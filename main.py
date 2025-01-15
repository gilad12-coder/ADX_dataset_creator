"""
A script to:
1. Fetch data from Azure Data Explorer (ADX) using a specified time period (day/week/month).
2. Sample data using a selection ratio.
3. Optionally reduce dimensionality using PCA.
4. Cluster data using HDBSCAN with FAISS acceleration.
5. Sample proportionally from each cluster based on cluster sizes.
6. Concatenate all periods into a final compressed dataset.
"""

import logging
from ADX_dataset_creation import compress_dataset
from config import (
    start_date,
    end_date,
    selection_ratio,
    vector_column,
    time_period,
    use_pca,
    pca_components,
    min_cluster_size,
    n_neighbors_faiss,
    connection_string,
    database,
    table_name,
    aad_app_id,
    app_key,
    authority_id
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_compression.log'),
        logging.StreamHandler()
    ]
)

def main():
    """
    Main function:
    1. Uses ADX authentication credentials from config
    2. Processes data using specified time period granularity
    3. Samples data based on selection ratio
    4. Reduces dimensionality with PCA if enabled
    5. Clusters using HDBSCAN+FAISS and samples proportionally
    6. Saves final compressed DataFrame as pickle file
    """
    logger.info(f"Processing data from {start_date} to {end_date}")
    compressed_df = compress_dataset(
        connection_string=connection_string,
        aad_app_id=aad_app_id,
        app_key=app_key,
        authority_id=authority_id,
        database=database,
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
        time_period=time_period,
        selection_ratio=selection_ratio,
        vector_column=vector_column,
        use_pca=use_pca,
        pca_components=pca_components,
        min_cluster_size=min_cluster_size,
        n_neighbors_faiss=n_neighbors_faiss,
        logger=logger
    )
    if compressed_df.empty:
        logger.warning("Compressed dataset is empty!")
    logger.info(f"Compressed dataset shape: {compressed_df.shape}")
    logger.info("\nSample of compressed dataset:")
    logger.info(f"\n{compressed_df.head()}")
    output_file = "compressed_data.pkl"
    logger.info(f"Saving compressed dataset to {output_file}")
    compressed_df.to_pickle(output_file)
    logger.info("Dataset compression completed successfully!")

if __name__ == "__main__":
    main()