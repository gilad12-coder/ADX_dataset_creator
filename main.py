"""
A script to:
1. Fetch data from Azure Data Explorer (ADX) day by day.
2. Randomly sample daily data.
3. Optionally reduce dimensionality (PCA).
4. Cluster data using either K-Means or HDBSCAN+FAISS (for faster neighbor lookups).
5. Sample an equal number of rows from each cluster.
6. Concatenate the final 'compressed' dataset.
"""
from ADX_dataset_creation import compress_dataset
from ADX_dataset_creation import create_kusto_client
from config import start_date, end_date, daily_sample_size, vector_column, clustering_method, num_clusters, samples_per_cluster, use_pca, pca_components, min_cluster_size, n_neighbors_faiss, cluster_url, tenant_id, database, table_name

def main():
    """
    Main function:
    1. Creates a KustoClient
    2. Defines parameters (dates, daily sample size, cluster settings, etc.)
    3. Calls compress_dataset() to fetch, cluster, and compress the dataset
    4. Prints or saves the final compressed DataFrame
    """
    kusto_client = create_kusto_client(cluster_url, tenant_id)

    compressed_df = compress_dataset(
        kusto_client=kusto_client,
        database=database,
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
        daily_sample_size=daily_sample_size,
        vector_column=vector_column,
        samples_per_cluster=samples_per_cluster,
        clustering_method=clustering_method,
        num_clusters=num_clusters,
        use_pca=use_pca,
        pca_components=pca_components,
        min_cluster_size=min_cluster_size,
        n_neighbors_faiss=n_neighbors_faiss
    )

    print(f"Compressed dataset shape: {compressed_df.shape}")
    print("Sample of compressed dataset:")
    print(compressed_df.head())
    compressed_df.to_pickle("compressed_data.pkl", index=False)