# Databricks notebook source
# Core Libraries
import pandas as pd
import numpy as np

# Database Connection
import hdbcli.dbapi as dbapi

# Generative AI for natural language processing
from langchain import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI


# COMMAND ----------

ocel_file = "/Users/I552639/Downloads/BPI Challenge 2019 (OCEL)_1_all/BPIC19.json"  

with open(ocel_file, "r", encoding="utf-8") as f:
    ocel = json.load(f)    

# COMMAND ----------

# Flatten JSON structure (each event contains multiple objects)
rows = []
for e_id, event in ocel["ocel:events"].items():
    timestamp = event.get("ocel:timestamp")
    activity = event.get("ocel:activity")
    for obj in event.get("ocel:omap", []):
        rows.append({
            "event_id": e_id,
            "object_id": obj,
            "activity": activity,
            "timestamp": timestamp
        })

df = pd.DataFrame(rows)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp', 'activity'])

# -----------------------------------------------------------
# Step 2: Compute Case-Level Metrics (per object_id)
# -----------------------------------------------------------
case_metrics = (
    df.groupby('object_id')
      .agg(start_time=('timestamp', 'min'),
           end_time=('timestamp', 'max'),
           activity_count=('activity', 'count'))
      .reset_index()
)

# Calculate cycle time in hours
case_metrics['cycle_time_hours'] = (case_metrics['end_time'] - case_metrics['start_time']).dt.total_seconds() / 3600

# -----------------------------------------------------------
# Step 3: Compute Benchmark Percentiles Automatically
# -----------------------------------------------------------
# Percentile ranks (0–100)
case_metrics['cycle_time_percentile'] = case_metrics['cycle_time_hours'].rank(pct=True) * 100
case_metrics['activity_count_percentile'] = case_metrics['activity_count'].rank(pct=True) * 100

# Create performance tiers
case_metrics['performance_tier'] = pd.cut(
    case_metrics['cycle_time_percentile'],
    bins=[0, 25, 50, 75, 100],
    labels=['Top Performer', 'Above Average', 'Below Average', 'Low Performer']
)

# -----------------------------------------------------------
# Step 4: Generate Benchmark Summary
# -----------------------------------------------------------
benchmark_summary = {
    'mean_cycle_time': case_metrics['cycle_time_hours'].mean(),
    'median_cycle_time': case_metrics['cycle_time_hours'].median(),
    'best_10_percent': np.percentile(case_metrics['cycle_time_hours'], 10),
    'worst_10_percent': np.percentile(case_metrics['cycle_time_hours'], 90)
}

print("\nBenchmark Summary Statistics:")
for k, v in benchmark_summary.items():
    print(f"{k:25s}: {v:.2f}")

print("\nSample Benchmark Percentiles:")
print(case_metrics.head(10))

# -----------------------------------------------------------
# Step 5: Store Benchmark Data to SAP HANA
# -----------------------------------------------------------
hana_host = "hana-host"
hana_port = 30015
hana_user = "hana-username"
hana_pass = "hana-password"

conn = dbapi.connect(
    address=hana_host,
    port=hana_port,
    user=hana_user,
    password=hana_pass
)
cur = conn.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS BENCHMARK_PERCENTILES (
    OBJECT_ID NVARCHAR(100),
    START_TIME TIMESTAMP,
    END_TIME TIMESTAMP,
    ACTIVITY_COUNT INT,
    CYCLE_TIME_HOURS DOUBLE,
    CYCLE_TIME_PERCENTILE DOUBLE,
    ACTIVITY_COUNT_PERCENTILE DOUBLE,
    PERFORMANCE_TIER NVARCHAR(30)
)
""")

# Insert calculated benchmark data
insert_sql = """
INSERT INTO BENCHMARK_PERCENTILES
(OBJECT_ID, START_TIME, END_TIME, ACTIVITY_COUNT, CYCLE_TIME_HOURS,
 CYCLE_TIME_PERCENTILE, ACTIVITY_COUNT_PERCENTILE, PERFORMANCE_TIER)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

for _, row in case_metrics.iterrows():
    cur.execute(insert_sql, (
        str(row['object_id']),
        row['start_time'].to_pydatetime(),
        row['end_time'].to_pydatetime(),
        int(row['activity_count']),
        float(row['cycle_time_hours']),
        float(row['cycle_time_percentile']),
        float(row['activity_count_percentile']),
        str(row['performance_tier'])
    ))

conn.commit()
cur.close()
conn.close()

print('Benchmark data successfully written to SAP HANA!")