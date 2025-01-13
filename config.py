import datetime

# Date range settings
start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2024, 1, 3)

# Data fetching settings
daily_sample_size = 50
vector_column = "EmbeddingVector"

# Clustering settings
clustering_method = "hdbscan"
num_clusters = 5
total_daily_samples = 25
use_pca = True
pca_components = 10
min_cluster_size = 5
n_neighbors_faiss = 15

# Azure Data Explorer settings
cluster_url = "https://<YourClusterName>.<region>.kusto.windows.net"
tenant_id = "<YourTenantID>"
database = "<YourDatabase>"
table_name = "<YourTable>"
