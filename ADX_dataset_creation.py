import datetime
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import faiss
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

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
        cluster_url, 
        authority_id=tenant_id
    )
    return KustoClient(kcsb)


def fetch_data_for_day(
    kusto_client: KustoClient, 
    database: str, 
    table_name: str, 
    day: datetime.date, 
    random_sample_size: int
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
    # Adjust your query to match your schema, including date/time fields.
    # Example assumes a column "Timestamp" is available, and that vector column is present.
    query = f"""
    {table_name}
    | where Timestamp >= datetime({day.isoformat()}) 
        and Timestamp < datetime({(day + datetime.timedelta(days=1)).isoformat()})
    | sample {random_sample_size}
    """
    properties = ClientRequestProperties()
    response = kusto_client.execute(database, query, properties=properties)
    df = dataframe_from_result_table(response.primary_results[0])
    
    return df


def cluster_data_kmeans(
    vectors: np.ndarray,
    num_clusters: int = 5,
    random_state: int = 42
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
    cluster_labels = kmeans.fit_predict(vectors)
    return cluster_labels


def cluster_data_hdbscan_faiss(
    vectors: np.ndarray,
    min_cluster_size: int = 5,
    n_neighbors: int = 15
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

    # Ensure float32 for FAISS indexing
    vectors_32 = vectors.astype(np.float32)

    # Build a FlatL2 index (exact search). For large datasets, consider
    # other faiss indexes (e.g., IVF) for approximate searching.
    index = faiss.IndexFlatL2(n_dim)
    index.add(vectors_32)  # add all vectors to index

    # For each vector, find the top n_neighbors
    # distances: shape (n_samples, n_neighbors)
    # indices: shape (n_samples, n_neighbors)
    distances, indices = index.search(vectors_32, k=n_neighbors)

    # Build a condensed distance matrix in the format HDBSCAN expects.
    # We'll do a simple approach:
    # 1) Create a large adjacency list (row i -> neighbors -> distances).
    # 2) We'll fill in a precomputed distance matrix:
    #    - For points that are not direct neighbors, distance can be large or inf.
    # 
    # The final shape must be (n_samples, n_samples) for 'precomputed' metric,
    # or we can pass a condensed distance matrix. But that is O(n^2).
    # 
    # For large n, building the full n x n matrix is huge. Instead, HDBSCAN
    # also allows 'precomputed' with a sparse distance matrix. We can pass
    # a 'DistanceMetric' that references a custom neighbor graph. 
    # 
    # => As a simple approach here (for demonstration), we'll build a dense matrix,
    #    which is feasible for smaller n. For truly large n, consider a sparse approach.

    distance_matrix = np.full((n_samples, n_samples), np.inf, dtype=np.float32)

    # Distance to self = 0
    np.fill_diagonal(distance_matrix, 0.0)

    # Fill in neighbor distances
    for i in range(n_samples):
        for j_idx in range(1, n_neighbors):  # neighbor 0 is the point itself
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # symmetric

    # HDBSCAN in 'precomputed' mode
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size
    )
    labels = clusterer.fit_predict(distance_matrix)

    return labels


def cluster_and_sample(
    df: pd.DataFrame,
    vector_column: str,
    total_samples: int = 1000,
    clustering_method: str = 'kmeans',
    num_clusters: int = 5,
    use_pca: bool = True,
    pca_components: int = 50,
    min_cluster_size: int = 5,
    n_neighbors_faiss: int = 15
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
    # Convert vectors from e.g. string or list to numpy array
    vectors = np.array(df[vector_column].tolist())
    
    # Optional PCA dimensionality reduction
    if use_pca and pca_components > 0 and pca_components < vectors.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        vectors = pca.fit_transform(vectors)
    
    # Perform clustering
    if clustering_method.lower() == 'kmeans':
        cluster_labels = cluster_data_kmeans(vectors, num_clusters=num_clusters)
    elif clustering_method.lower() == 'hdbscan':
        cluster_labels = cluster_data_hdbscan_faiss(
            vectors,
            min_cluster_size=min_cluster_size,
            n_neighbors=n_neighbors_faiss
        )
    else:
        raise ValueError("clustering_method must be either 'kmeans' or 'hdbscan'.")
        
    df['cluster_label'] = cluster_labels
    
    # HDBSCAN can produce -1 for outliers
    if clustering_method.lower() == 'hdbscan':
        # Filter out outliers if cluster_label == -1
        df = df[df['cluster_label'] != -1]
        
    if df.empty:
        return df  # No data left after outlier removal or no valid clusters
    
    # Calculate cluster sizes and proportions
    cluster_sizes = df['cluster_label'].value_counts()
    total_points = cluster_sizes.sum()
    cluster_proportions = cluster_sizes / total_points
    
    # Calculate number of samples per cluster
    samples_per_cluster = (cluster_proportions * total_samples).round().astype(int)
    
    # Adjust for rounding errors to ensure we get exactly total_samples
    diff = total_samples - samples_per_cluster.sum()
    if diff != 0:
        # Add/subtract the difference from the largest cluster(s)
        indices_to_adjust = cluster_sizes.nlargest(abs(diff)).index
        for idx in indices_to_adjust:
            samples_per_cluster[idx] += np.sign(diff)
    
    # Collect proportional samples from each cluster
    final_rows = []
    for cluster_id, n_samples in samples_per_cluster.items():
        cluster_df = df[df['cluster_label'] == cluster_id]
        
        # If cluster is smaller than desired samples, take all points
        if len(cluster_df) <= n_samples:
            final_rows.append(cluster_df)
        else:
            final_rows.append(cluster_df.sample(n_samples, random_state=42))
    
    # Concatenate samples
    final_df = pd.concat(final_rows, ignore_index=True)
    
    # Drop the temporary cluster_label column
    final_df.drop(columns=['cluster_label'], inplace=True)
    
    return final_df


def compress_dataset(
    kusto_client: KustoClient,
    database: str,
    table_name: str,
    start_date: datetime.date,
    end_date: datetime.date,
    daily_sample_size: int,
    vector_column: str,
    samples_per_cluster: int,
    clustering_method: str,
    num_clusters: int,
    use_pca: bool,
    pca_components: int,
    min_cluster_size: int,
    n_neighbors_faiss: int
) -> pd.DataFrame:
    """
    Fetches data from ADX between start_date and end_date, day by day.
    Clusters the daily data (via K-Means or HDBSCAN+FAISS), samples equally
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
    samples_per_cluster : int
        How many rows to sample from each cluster per day.
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

    # Iterate from start_date to end_date (inclusive) day by day
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        day = dt.date()
        print(f"Processing day: {day.isoformat()}")

        # Step 1: Fetch a random sample from ADX for this day
        df_day = fetch_data_for_day(
            kusto_client=kusto_client,
            database=database,
            table_name=table_name,
            day=day,
            random_sample_size=daily_sample_size
        )

        if df_day.empty:
            print(f"No data returned for {day.isoformat()}. Skipping.")
            continue

        # Step 2: Cluster the daily data and sample equally from each cluster
        df_clustered = cluster_and_sample(
            df=df_day,
            vector_column=vector_column,
            clustering_method=clustering_method,
            num_clusters=num_clusters,
            samples_per_cluster=samples_per_cluster,
            use_pca=use_pca,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            n_neighbors_faiss=n_neighbors_faiss
        )

        if df_clustered.empty:
            print(f"All data for {day.isoformat()} was outlier/no data post-clustering.")
            continue

        final_data.append(df_clustered)

    # Concatenate final list of daily cluster-sampled data
    if not final_data:
        print("No data was collected over the entire date range. Returning empty DataFrame.")
        return pd.DataFrame()

    compressed_df = pd.concat(final_data, ignore_index=True)
    return compressed_df
