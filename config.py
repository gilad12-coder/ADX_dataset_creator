import datetime

start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2024, 1, 3)
daily_sample_size = 50
vector_column = "EmbeddingVector"
clustering_method = "hdbscan"
num_clusters = 5
samples_per_cluster = 5
use_pca = True
pca_components = 10
min_cluster_size = 5
n_neighbors_faiss = 15
cluster_url = "https://<YourClusterName>.<region>.kusto.windows.net"
tenant_id = "<YourTenantID>"
database = "<YourDatabase>"
table_name = "<YourTable>"