from enum import Enum

class TimePeriod(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

CLUSTER_LABEL_COLUMN = "cluster_label"
TIMESTAMP_COLUMN = "Timestamp"

QUERY_TEMPLATE = """{table_name}
    | where {TIMESTAMP_COLUMN} >= datetime({start_date}) 
        and {TIMESTAMP_COLUMN} < datetime({end_date})
    | sample {selection_ratio}"""