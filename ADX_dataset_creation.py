import datetime
from cons import TimePeriod, CLUSTER_LABEL_COLUMN, TIMESTAMP_COLUMN, QUERY_TEMPLATE
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import hdbscan
import faiss
from azure.kusto.data import (
    KustoClient,
    KustoConnectionStringBuilder,
    ClientRequestProperties,
)
from azure.kusto.data.helpers import dataframe_from_result_table
import logging

def create_kusto_client(connection_string: str, aad_app_id: str, app_key: str, authority_id: str) -> KustoClient:
    """
    Creates an authenticated KustoClient instance for connecting to Azure Data Explorer (Kusto).

    This function establishes a connection using AAD application key authentication with the provided
    credentials and connection details.

    Parameters:
        connection_string (str): The Kusto cluster connection string in the format
            'https://<cluster_name>.<region>.kusto.windows.net'
        aad_app_id (str): The Azure Active Directory application (client) ID
        app_key (str): The application's secret key or client secret
        authority_id (str): The Azure AD tenant ID where the application is registered

    Returns:
        KustoClient: An authenticated client instance that can be used to execute queries
    """
    kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
        connection_string=connection_string,
        aad_app_id=aad_app_id,
        app_key=app_key,
        authority_id=authority_id
    )
    return KustoClient(kcsb)

def fetch_data_for_period(
    kusto_client: KustoClient,
    database: str,
    table_name: str,
    start_date: datetime.date,
    period: TimePeriod,
    selection_ratio: float,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Fetches data from Azure Data Explorer for a specific time period with initial sampling.

    Parameters:
        kusto_client (KustoClient): The KustoClient to execute the query
        database (str): The name of the database in ADX
        table_name (str): The name of the table to query
        start_date (datetime.date): The start date of the period.
        period (TimePeriod): The time period to fetch (day, week, or month).
        selection_ratio (float): Ratio to sample the data (e.g., 0.1 for 10%)
        logger (logging.Logger) A logger to log statements.

    Returns:
        pd.DataFrame: A DataFrame with the sampled data for the specified period
    """
    if period == TimePeriod.DAY:
        end_date = start_date + datetime.timedelta(days=1)
    elif period == TimePeriod.WEEK:
        end_date = start_date + datetime.timedelta(weeks=1)
    else:  # MONTH
        if start_date.month == 12:
            end_date = datetime.date(start_date.year + 1, 1, 1)
        else:
            end_date = datetime.date(start_date.year, start_date.month + 1, 1)
    query = QUERY_TEMPLATE.format(
        table_name=table_name,
        TIMESTAMP_COLUMN=TIMESTAMP_COLUMN,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        selection_ratio=selection_ratio
    )
    properties = ClientRequestProperties()
    response = kusto_client.execute(database, query, properties=properties)
    df = dataframe_from_result_table(response.primary_results[0])
    logger.info(f"Retrieved {len(df)} rows for period {start_date} to {end_date}")
    return df


def cluster_data_hdbscan_faiss(
    vectors: np.ndarray,
    min_cluster_size: int = 5,
    n_neighbors: int = 15
) -> np.ndarray:
    """
    Clusters high-dimensional data using HDBSCAN with FAISS-accelerated nearest neighbor search.

    This function performs density-based clustering using HDBSCAN algorithm, while leveraging
    FAISS for efficient nearest neighbor computation. It works by:
    1. Converting vectors to float32 for FAISS compatibility
    2. Building a FAISS index for fast nearest neighbor search
    3. Computing a distance matrix using k-nearest neighbors
    4. Applying HDBSCAN clustering on the precomputed distance matrix

    Parameters:
        vectors (np.ndarray): Input data matrix of shape (n_samples, n_features) where each
            row represents a vector to be clustered
        min_cluster_size (int, optional): The minimum size of clusters. Smaller clusters
            will be labeled as noise (-1). Defaults to 5.
        n_neighbors (int, optional): Number of nearest neighbors to consider when building
            the distance matrix. Defaults to 15.

    Returns:
        np.ndarray: Array of cluster labels of shape (n_samples).
    """
    n_samples, n_dim = vectors.shape
    vectors_32 = vectors.astype(np.float32)
    index = faiss.IndexFlatL2(n_dim)
    index.add(vectors_32)
    distances = np.empty((n_samples, n_neighbors), dtype=np.float32)
    indices = np.empty((n_samples, n_neighbors), dtype=np.int64)
    index.search(vectors_32, n_neighbors, distances, indices)
    distance_matrix = np.full((n_samples, n_samples), np.inf, dtype=np.float32)
    np.fill_diagonal(distance_matrix, 0.0)
    for i in range(n_samples):
        for j_idx in range(1, n_neighbors):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        algorithm='generic'
    )
    return clusterer.fit_predict(distance_matrix)


def cluster_and_sample_period(
    df: pd.DataFrame,
    vector_column: str,
    logger: logging.Logger,
    use_pca: bool = True,
    pca_components: int = 50,
    min_cluster_size: int = 5,
    n_neighbors_faiss: int = 15
) -> pd.DataFrame:
    """
    Clusters the data using HDBSCAN and samples proportionally to cluster sizes.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the vectors
        vector_column (str): Column name containing vector data
        logger (logging.Logger) A logger to log statements.
        use_pca (bool): Whether to use PCA
        pca_components (int): Number of PCA components
        min_cluster_size (int): Minimum cluster size for HDBSCAN
        n_neighbors_faiss (int): Number of neighbors for FAISS

    Returns:
        pd.DataFrame: DataFrame with proportionally sampled data from clusters
    """
    if df.empty:
        return df
    vectors = np.array(df[vector_column].tolist())
    if use_pca and 0 < pca_components < vectors.shape[1]:
        logger.debug(f"Applying PCA reduction to {vectors.shape[1]} dimensions")
        pca = PCA(n_components=pca_components, random_state=42)
        logger.info(f"Reduced dimensions to {vectors.shape[1]} components")
        vectors = pca.fit_transform(vectors)
    logger.debug("Starting HDBSCAN clustering")
    cluster_labels = cluster_data_hdbscan_faiss(
        vectors,
        min_cluster_size=min_cluster_size,
        n_neighbors=n_neighbors_faiss
    )
    df = df.copy()
    df[CLUSTER_LABEL_COLUMN] = cluster_labels
    df = df[df[CLUSTER_LABEL_COLUMN] != -1]
    if df.empty:
        logger.warning("No data points remained after removing noise")
        return df
    # Calculate cluster sizes and proportions
    cluster_sizes = df[CLUSTER_LABEL_COLUMN].value_counts()
    total_points = cluster_sizes.sum()
    cluster_proportions = cluster_sizes / total_points
    # Sample from each cluster according to its proportion
    sampled_dfs = []
    for cluster_id in cluster_sizes.index:
        cluster_df = df[df[CLUSTER_LABEL_COLUMN] == cluster_id]
        cluster_size = len(cluster_df)
        # Calculate how many samples we should take from this cluster
        samples_from_cluster = int(np.round(cluster_proportions[cluster_id] * total_points))
        if samples_from_cluster > 0:
            # If we need more samples than we have, take all available
            if samples_from_cluster >= cluster_size:
                sampled_dfs.append(cluster_df)
            else:
                sampled_dfs.append(cluster_df.sample(
                    n=samples_from_cluster,
                    random_state=42
                ))
            logger.debug(f"Cluster {cluster_id}: sampled {samples_from_cluster} from {cluster_size} points")
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    final_df.drop(columns=[CLUSTER_LABEL_COLUMN], inplace=True)
    logger.info(f"Final sampled dataset size: {len(final_df)}")
    return final_df


def compress_dataset(
    connection_string: str,
    aad_app_id: str,
    app_key: str,
    authority_id: str,
    database: str,
    table_name: str,
    start_date: datetime.date,
    end_date: datetime.date,
    time_period: TimePeriod,
    selection_ratio: float,
    vector_column: str,
    logger: logging.Logger,
    use_pca: bool = True,
    pca_components: int = 50,
    min_cluster_size: int = 5,
    n_neighbors_faiss: int = 15
) -> pd.DataFrame:
    """
    Process data over a time range using specified period granularity.

    Parameters:
        connection_string (str): The Kusto cluster connection string
        aad_app_id (str): The Azure Active Directory application ID
        app_key (str): The application key/secret
        authority_id (str): The Azure AD tenant ID
        database (str): Database name
        table_name (str): Table name
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        time_period (TimePeriod): Day, Week, or Month
        selection_ratio (float): Ratio for initial data sampling
        vector_column (str): Column containing vector data
        use_pca (bool): Whether to use PCA
        pca_components (int): Number of PCA components
        min_cluster_size (int): Minimum cluster size for HDBSCAN
        n_neighbors_faiss (int): Number of neighbors for FAISS
        logger (logging.Logger) A logger to log statements.

    Returns:
        pd.DataFrame: Processed and clustered data
    """
    kusto_client = create_kusto_client(
        connection_string=connection_string,
        aad_app_id=aad_app_id,
        app_key=app_key,
        authority_id=authority_id
    )
    final_data = []
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"Processing period starting: {current_date.isoformat()}")
        df_period = fetch_data_for_period(
            kusto_client=kusto_client,
            database=database,
            table_name=table_name,
            start_date=current_date,
            period=time_period,
            selection_ratio=selection_ratio,
            logger=logger
        )
        if not df_period.empty:
            df_clustered = cluster_and_sample_period(
                df=df_period,
                vector_column=vector_column,
                use_pca=use_pca,
                pca_components=pca_components,
                min_cluster_size=min_cluster_size,
                n_neighbors_faiss=n_neighbors_faiss,
                logger=logger
            )
            if not df_clustered.empty:
                final_data.append(df_clustered)
        # Advance to next period
        if time_period == TimePeriod.DAY:
            current_date += datetime.timedelta(days=1)
        elif time_period == TimePeriod.WEEK:
            current_date += datetime.timedelta(weeks=1)
        else:  # MONTH
            if current_date.month == 12:
                current_date = datetime.date(current_date.year + 1, 1, 1)
            else:
                current_date = datetime.date(current_date.year, current_date.month + 1, 1)
    if not final_data:
        logger.warning("No data was collected over the entire date range.")
        return pd.DataFrame()
    return pd.concat(final_data, ignore_index=True)