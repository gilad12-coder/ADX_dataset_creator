import datetime
from cons import TimePeriod

# Date range settings
start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2024, 1, 3)

# Data sampling and processing settings
selection_ratio = 0.1
vector_column = "EmbeddingVector"
time_period = TimePeriod.DAY

# Dimensionality reduction and clustering settings
use_pca = True
pca_components = 10
min_cluster_size = 5
n_neighbors_faiss = 15

# Azure Data Explorer settings
connection_string = "https://<YourClusterName>.<region>.kusto.windows.net"
database = "<YourDatabase>"
table_name = "<YourTable>"
aad_app_id = "<YourAADAppID>"
app_key = "<YourAppKey>"
authority_id = "<YourTenantID>"