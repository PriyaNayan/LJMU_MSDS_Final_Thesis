# Databricks notebook source
# MAGIC %pip install requests hana-ml hdbcli generative-ai-hub-sdk[all]==4.4.3 
# MAGIC %pip install hdbcli sqlalchemy-hana

# COMMAND ----------

from langchain_community.utilities import SQLDatabase
import json
import os
from pyspark.sql import functions as F

# COMMAND ----------

# Create Generative AI SQL Agent with Custom Prompt

print("\n Initializing Generative AI SQL Agent...")

connection_uri = f"hana+pyhdb://{hana_user}:{password*****}@{hana_host}:{hana_port}/{hana_schema}"
db = SQLDatabase.from_uri(connection_uri)
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

# Custom prompt to guide the agent
CUSTOM_SYSTEM_PROMPT = """
You are an intelligent Business Process Benchmarking Assistant.

You are connected to a SAP HANA database that contains the table BENCHMARK_PERCENTILES.
The table has the following columns:
- OBJECT_ID (NVARCHAR)
- START_TIME (TIMESTAMP)
- END_TIME (TIMESTAMP)
- ACTIVITY_COUNT (INT)
- CYCLE_TIME_HOURS (DOUBLE)
- CYCLE_TIME_PERCENTILE (DOUBLE)
- ACTIVITY_COUNT_PERCENTILE (DOUBLE)
- PERFORMANCE_TIER (NVARCHAR)

Your goals:
1. Generate correct and optimized SQL queries to answer user questions.
2. Base all reasoning only on this table's data.
3. Summarize findings in clear, human-friendly language.
4. Highlight top performers, bottlenecks, and improvement areas when relevant.
5. NEVER make up columns or use tables not in the schema.
6. If the question is ambiguous, clarify assumptions before executing SQL.
"""

# Wrap custom prompt as a LangChain System Message
system_message = SystemMessagePromptTemplate.from_template(CUSTOM_SYSTEM_PROMPT)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
    system_message=SystemMessage(content=CUSTOM_SYSTEM_PROMPT)
)

print("SQL Agent ready for natural language business queries!")

# STEP 6: Example Business Queries

questions = [
    "Which object IDs are in the top 10 percent based on cycle time percentile?",
    "Summarize the number of cases in each performance tier.",
    "What is the average cycle time of the top performers compared to the low performers?",
    "List the top 5 slowest cases by cycle time and their activity counts.",
    "Provide a business insight on which segment requires process improvement."
]

for q in questions:
    print("\n Business Question:", q)
    try:
        response = agent.run(q)
        print(" Answer:\n", response)
    except Exception as e:
        print(" Error:", e)

print("\nEnd-to-End Execution Complete!")