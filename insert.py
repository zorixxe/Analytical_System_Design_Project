import time
from cassandra.cluster import Cluster
import json

# Retry connection logic
def connect_to_cassandra(max_retries=30, retry_delay=5):
    for i in range(max_retries):
        try:
            cluster = Cluster(["cassandra_db"])  # Use the container name
            session = cluster.connect()
            print("Successfully connected to Cassandra!")
            return session
        except Exception as e:
            print(f"Cassandra not ready yet... retrying ({i+1}/{max_retries})")
            time.sleep(retry_delay)
    
    print("❌ Failed to connect to Cassandra after multiple attempts. Exiting.")
    exit()

# Connect to Cassandra
session = connect_to_cassandra()

# Create keyspace
session.execute("""
    CREATE KEYSPACE IF NOT EXISTS inlo_data 
    WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
""")

# Set keyspace
session.set_keyspace("inlo_data")

# Create table to store all statistics
session.execute("""
    CREATE TABLE IF NOT EXISTS kommun_data (
        kommun_code TEXT,
        year INT,
        income DOUBLE,
        avg_size DOUBLE,
        population INT,
        PRIMARY KEY (kommun_code, year)
    );
""")

# Load JSON data
with open("./data/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Dictionary to store values before inserting into Cassandra
kommun_records = {}

for entry in data["data"]:
    year = int(entry["key"][0])  # Extract year
    kommun_code = entry["key"][1]  # Municipality code
    data_type = entry["key"][2]  # Data type (uppgifter)
    
    if (kommun_code, year) not in kommun_records:
        kommun_records[(kommun_code, year)] = {"income": None, "avg_size": None, "population": None}

    if data_type == "0":  # Population of household-dwelling units
        kommun_records[(kommun_code, year)]["population"] = int(entry["values"][0])
    elif data_type == "2":  # Average size of household-dwelling units
        kommun_records[(kommun_code, year)]["avg_size"] = float(entry["values"][0])
    elif data_type == "4":  # Disposable monetary income of household-dwelling units, mean
        kommun_records[(kommun_code, year)]["income"] = float(entry["values"][0])

# Insert data into Cassandra
batch_query = session.prepare("""
    INSERT INTO kommun_data (kommun_code, year, income, avg_size, population)
    VALUES (?, ?, ?, ?, ?)
""")

for (kommun_code, year), values in kommun_records.items():
    session.execute(batch_query, (
        kommun_code, year,
        values["income"] if values["income"] is not None else 0.0,  # Handle missing values
        values["avg_size"] if values["avg_size"] is not None else 0.0,
        values["population"] if values["population"] is not None else 0
    ))
    print(f"Inserted: kommun_code={kommun_code}, year={year}, income={values['income']}, avg_size={values['avg_size']}, population={values['population']}")

print("Data inserted into Cassandra successfully!")
