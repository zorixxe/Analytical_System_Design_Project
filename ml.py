import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ETL-Pipeline") \
    .config("spark.jars", "/opt/bitnami/spark/jars/spark-cassandra-connector_2.12-3.2.0.jar") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
    .config("spark.cassandra.connection.host", "cassandra_db") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

# Extract Data (Read from Cassandra)
def extract_data():
    for i in range(20):  # Retry up to 20 times
        try:
            df = spark.read.format("org.apache.spark.sql.cassandra").options(
                table="kommun_data", keyspace="inlo_data").load()
            if df.count() > 0:
                return df
        except Exception as e:
            print(f"Waiting for Cassandra... ({i+1}/20)")
    print("Cassandra is not available. Exiting.")
    exit()

# Transform Data (Convert to Pandas for ML Processing)
def transform_data(spark_df):
    try:
        spark_df.show(n=20, truncate=False)
        pandas_df = spark_df.toPandas()
        return pandas_df
    except Exception as e:
        print(f"Error transforming data: {e}")
        exit()

if __name__ == "__main__":
    # Extract and transform data
    spark_df = extract_data()
    df = transform_data(spark_df)

    # Ensure dataframe has necessary columns
    required_columns = {'kommun_code', 'year', 'income', 'population', 'avg_size'}
    if df.empty or not required_columns.issubset(df.columns):
        print(f"Error: Missing required data columns. Required columns: {required_columns}")
        exit()

    # Fetch kommun_code from environment variable
    kommun_code = os.getenv("KOMMUN_CODE", "all")  # Default to "all"
    print(f"🔍 Selected kommun_code: {kommun_code}")

    # Determine which kommun_codes to process
    kommun_codes_to_process = df['kommun_code'].unique() if kommun_code.lower() == 'all' else [kommun_code]

    # Store predictions for all kommun_codes
    all_predictions = []

    for kommun in kommun_codes_to_process:
        kommun_df = df[df['kommun_code'] == kommun]

        if kommun_df.shape[0] < 2:
            print(f"Not enough data for kommun_code {kommun}. Skipping.")
            continue

        # Prepare features and target variable
        X = kommun_df[['year', 'population', 'avg_size']]
        y = kommun_df['income']

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Prepare future data for prediction (next 5 years)
        future_years = np.arange(2024, 2029)
        future_population = kommun_df['population'].iloc[-1] * (1.01 ** np.arange(1, 6))  # Assume 1% growth
        future_avg_size = np.full(5, kommun_df['avg_size'].iloc[-1])

        future_data = pd.DataFrame({'year': future_years, 'population': future_population, 'avg_size': future_avg_size})
        future_data['kommun_code'] = kommun  # Add kommun_code for identification

        # Predict future income
        predicted_income = model.predict(future_data[['year', 'population', 'avg_size']])
        future_data['predicted_income'] = predicted_income

        # Store results for each kommun_code
        all_predictions.append(future_data)

        # Save plot
        plt.figure(figsize=(8, 5))
        plt.plot(kommun_df['year'], kommun_df['income'], marker='o', linestyle='-', label='Actual Income')
        plt.plot(future_data['year'], future_data['predicted_income'], marker='s', linestyle='dashed', color='red', label='Predicted Income')
        plt.xlabel('Year')
        plt.ylabel('Income')
        plt.title(f'Income Prediction for Kommun {kommun} (2024-2028)')
        plt.legend()
        plt.grid()
        plt.savefig(f"/app/prediction_{kommun}.png")
        print(f"Prediction plot saved as /app/prediction_{kommun}.png")

    # Combine all predictions into one DataFrame
    final_predictions = pd.concat(all_predictions, ignore_index=True)

    # Print the predictions for all kommun_codes
    print("\n📊 Final Predictions:")
    print(final_predictions)
    print("\nAll predictions have been saved as image files in the /app/ directory!")
    print("To copy them from the Docker container to your local machine, run:")
    print("docker cp python_pipeline:/app/prediction_<komun_dode>.png .")
    print("start prediction_<komun_dode>.png")

