import datetime
from dateutil import rrule
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import faiss
from azure.kusto.data import (
    KustoClient,
    KustoConnectionStringBuilder,
    ClientRequestProperties,
)
from azure.kusto.data.helpers import dataframe_from_result_table


def create_kusto_client(cluster_url: str, tenant_id: str) -> KustoClient:
    """
    Create a KustoClient for a given Azure Data Explorer (Kusto) cluster.

    Parameters
    ----------
    cluster_url : str
        The URL of the Azure Data Explorer cluster
        (e.g. "https://<clusterName>.<region>.kusto.windows.net")
    tenant_id : str
        The Azure tenant ID used for authentication.

    Returns
    -------
    KustoClient
        Authenticated KustoClient object to run queries.
    """
    kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(
        cluster_url, authority_id=tenant_id
    )
    return KustoClient(kcsb)


def fetch_data_for_day(
    kusto_client: KustoClient,
    database: str,
    table_name: str,
    day: datetime.date,
    random_sample_size: int,
) -> pd.DataFrame:
    """
    Fetches a random sample of data from Azure Data Explorer for a specific day.

    Parameters
    ----------
    kusto_client : KustoClient
        The KustoClient to execute the query.
    database : str
        The name of the database in ADX.
    table_name : str
        The name of the table to query.
    day : datetime.date
        The day for which we want to fetch data.
    random_sample_size : int
        The number of rows to randomly sample.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the randomly sampled data for the specified day.
    """
    query = f"""{table_name}
    | where Timestamp >= datetime({day.isoformat()}) 
        and Timestamp < datetime({(day + datetime.timedelta(days=1)).isoformat()})
    | sample {random_sample_size}"""
    properties = ClientRequestProperties()
    response = kusto_client.execute(database, query, properties=properties)
    return dataframe_from_result_table(response.primary_results[0])


def cluster_data_kmeans(
    vectors: np.ndarray, num_clusters: int = 5, random_state: int = 42
) -> np.ndarray:
    """
    Clusters data using K-Means.

    Parameters
    ----------
    vectors : np.ndarray
        2D array of shape (n_samples, n_features).
    num_clusters : int, optional
        Number of clusters (K). Default 5.
    random_state : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    np.ndarray
        1D array of cluster labels of length n_samples.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    return kmeans.fit_predict(vectors)


def cluster_data_hdbscan_faiss(
    vectors: np.ndarray, min_cluster_size: int = 5, n_neighbors: int = 15
) -> np.ndarray:
    """
    Clusters data using HDBSCAN, using FAISS to accelerate neighbor searches.
    This function constructs a KNN distance graph and passes it in 'precomputed' mode.

    Parameters
    ----------
    vectors : np.ndarray
        2D array of shape (n_samples, n_features).
    min_cluster_size : int, optional
        The smallest size grouping that HDBSCAN will consider a cluster. Default 5.
    n_neighbors : int, optional
        Number of neighbors to consider for graph construction. Default 15.

    Returns
    -------
    np.ndarray
        1D array of cluster labels of length n_samples,
        or -1 for outliers (noise).
    """
    n_samples, n_dim = vectors.shape
    vectors_32 = vectors.astype(np.float32)
    index = faiss.IndexFlatL2(n_dim)
    index.add(vectors_32)
    distances, indices = index.search(vectors_32, k=n_neighbors)
    distance_matrix = np.full((n_samples, n_samples), np.inf, dtype=np.float32)
    np.fill_diagonal(distance_matrix, 0.0)
    for i in range(n_samples):
        for j_idx in range(1, n_neighbors):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(distance_matrix)


def cluster_and_sample(
    df: pd.DataFrame,
    vector_column: str,
    total_samples: int = 1000,
    clustering_method: str = "kmeans",
    num_clusters: int = 5,
    use_pca: bool = True,
    pca_components: int = 50,
    min_cluster_size: int = 5,
    n_neighbors_faiss: int = 15,
) -> pd.DataFrame:
    """
    Clusters the data based on a vector column using either K-Means or
    HDBSCAN (accelerated by FAISS), then samples rows proportionally to cluster sizes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the vectors.
    vector_column : str
        The column name which contains the vector data. Each entry in this column
        should be a list or array of floats.
    total_samples : int, optional
        Total number of samples to return across all clusters. Default is 1000.
    clustering_method : str, optional
        Which clustering algorithm to use, 'kmeans' or 'hdbscan'. Default is 'kmeans'.
    num_clusters : int, optional
        The number of clusters to use in K-Means. Ignored if clustering_method='hdbscan'.
        Default is 5.
    use_pca : bool, optional
        Whether to use PCA to reduce dimensionality before clustering. Default is True.
    pca_components : int, optional
        Number of PCA components to use if use_pca=True. Default is 50.
    min_cluster_size : int, optional
        The min_cluster_size parameter for HDBSCAN. Ignored for K-Means. Default is 5.
    n_neighbors_faiss : int, optional
        Number of neighbors to use when building the FAISS distance graph for HDBSCAN.
        Default is 15.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing samples from each cluster proportional to cluster sizes.
    """
    vectors = np.array(df[vector_column].tolist())
    if use_pca and 0 < pca_components < vectors.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        vectors = pca.fit_transform(vectors)
    if clustering_method.lower() == "kmeans":
        cluster_labels = cluster_data_kmeans(vectors, num_clusters=num_clusters)
    elif clustering_method.lower() == "hdbscan":
        cluster_labels = cluster_data_hdbscan_faiss(
            vectors, min_cluster_size=min_cluster_size, n_neighbors=n_neighbors_faiss
        )
    else:
        raise ValueError("clustering_method must be either 'kmeans' or 'hdbscan'.")
    df["cluster_label"] = cluster_labels
    if clustering_method.lower() == "hdbscan":
        df = df[df["cluster_label"] != -1]
    if df.empty:
        return df
    cluster_sizes = df["cluster_label"].value_counts()
    total_points = cluster_sizes.sum()
    cluster_proportions = cluster_sizes / total_points
    samples_per_cluster = (cluster_proportions * total_samples).round().astype(int)
    diff = total_samples - samples_per_cluster.sum()
    if diff != 0:
        indices_to_adjust = cluster_sizes.nlargest(abs(diff)).index
        for idx in indices_to_adjust:
            samples_per_cluster[idx] += np.sign(diff)
    final_rows = []
    for cluster_id, n_samples in samples_per_cluster.items():
        cluster_df = df[df["cluster_label"] == cluster_id]
        if len(cluster_df) <= n_samples:
            final_rows.append(cluster_df)
        else:
            final_rows.append(cluster_df.sample(n_samples, random_state=42))
    final_df = pd.concat(final_rows, ignore_index=True)
    final_df.drop(columns=["cluster_label"], inplace=True)
    return final_df


def compress_dataset(
    kusto_client: KustoClient,
    database: str,
    table_name: str,
    start_date: datetime.date,
    end_date: datetime.date,
    daily_sample_size: int,
    vector_column: str,
    total_daily_samples: int,
    clustering_method: str,
    num_clusters: int,
    use_pca: bool,
    pca_components: int,
    min_cluster_size: int,
    n_neighbors_faiss: int,
) -> pd.DataFrame:
    """
    Fetches data from ADX between start_date and end_date, day by day.
    Clusters the daily data (via K-Means or HDBSCAN+FAISS), samples proportionally
    from each cluster, then concatenates into a final compressed dataset.

    Parameters
    ----------
    kusto_client : KustoClient
        Authenticated KustoClient object to run ADX queries.
    database : str
        The name of the database in ADX.
    table_name : str
        The table name to query in ADX.
    start_date : datetime.date
        Start of the date range (inclusive).
    end_date : datetime.date
        End of the date range (inclusive).
    daily_sample_size : int
        How many rows to randomly sample each day before clustering.
    vector_column : str
        The name of the column that holds the vector data in the DataFrame.
    total_daily_samples : int
        Total number of samples to take per day, distributed proportionally across clusters.
    clustering_method : str
        Which clustering algorithm to use: 'kmeans' or 'hdbscan'.
    num_clusters : int
        Number of clusters (K) to use in K-Means. Ignored for HDBSCAN.
    use_pca : bool
        Whether to apply PCA for dimensionality reduction before clustering.
    pca_components : int
        Number of PCA components to use if use_pca=True.
    min_cluster_size : int
        The min_cluster_size parameter for HDBSCAN. Ignored for K-Means.
    n_neighbors_faiss : int
        Number of neighbors to consider for building the FAISS distance graph
        for HDBSCAN.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame of the compressed dataset across all days.
    """
    final_data = []
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        day = dt.date()
        print(f"Processing day: {day.isoformat()}")
        df_day = fetch_data_for_day(
            kusto_client=kusto_client,
            database=database,
            table_name=table_name,
            day=day,
            random_sample_size=daily_sample_size,
        )
        if df_day.empty:
            print(f"No data returned for {day.isoformat()}. Skipping.")
            continue
        df_clustered = cluster_and_sample(
            df=df_day,
            vector_column=vector_column,
            total_samples=total_daily_samples,
            clustering_method=clustering_method,
            num_clusters=num_clusters,
            use_pca=use_pca,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            n_neighbors_faiss=n_neighbors_faiss,
        )
        if df_clustered.empty:
            print(
                f"All data for {day.isoformat()} was outlier/no data post-clustering."
            )
            continue
        final_data.append(df_clustered)
    if not final_data:
        print(
            "No data was collected over the entire date range. Returning empty DataFrame."
        )
        return pd.DataFrame()
    return pd.concat(final_data, ignore_index=True)
