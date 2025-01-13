"""
A script to:
1. Fetch data from Azure Data Explorer (ADX) day by day.
2. Randomly sample daily data.
3. Optionally reduce dimensionality (PCA).
4. Cluster data using either K-Means or HDBSCAN+FAISS (for faster neighbor lookups).
5. Sample proportionally from each cluster based on cluster size.
6. Concatenate the final 'compressed' dataset.
"""
import pandas as pd
from clustering_module import compress_dataset, create_kusto_client
from config import (
    start_date, end_date, daily_sample_size, vector_column,
    clustering_method, num_clusters, total_daily_samples,
    use_pca, pca_components, min_cluster_size, n_neighbors_faiss,
    cluster_url, tenant_id, database, table_name
)

def main():
    """
    Main function:
    1. Creates a KustoClient
    2. Defines parameters (dates, daily sample size, cluster settings, etc.)
    3. Calls compress_dataset() to fetch, cluster, and compress the dataset
    4. Saves the final compressed DataFrame as a pickle file
    """
    try:
        print("Creating Kusto client...")
        kusto_client = create_kusto_client(cluster_url, tenant_id)
        
        print(f"Processing data from {start_date} to {end_date}")
        compressed_df = compress_dataset(
            kusto_client=kusto_client,
            database=database,
            table_name=table_name,
            start_date=start_date,
            end_date=end_date,
            daily_sample_size=daily_sample_size,
            vector_column=vector_column,
            total_daily_samples=total_daily_samples,
            clustering_method=clustering_method,
            num_clusters=num_clusters,
            use_pca=use_pca,
            pca_components=pca_components,
            min_cluster_size=min_cluster_size,
            n_neighbors_faiss=n_neighbors_faiss
        )
        
        if compressed_df.empty:
            print("Warning: Compressed dataset is empty!")
            return
            
        print(f"Compressed dataset shape: {compressed_df.shape}")
        print("\nSample of compressed dataset:")
        print(compressed_df.head())
        
        output_file = "compressed_data.pkl"
        print(f"\nSaving compressed dataset to {output_file}")
        compressed_df.to_pickle(output_file)
        print("Done!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
