# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
# Databricks Part 2 – Simulated Streaming with Model

# %%
#### Load the saved model
from pyspark.ml.pipeline import PipelineModel

model = PipelineModel.load("/dbfs/FileStore/models/fake_news_best_model")
print("** Model loaded.")

# %%
print(type(model))

# %%
#### Load the streaming data (new messages)
df_stream = spark.read.csv("/FileStore/tables/stream1.csv", header=True, inferSchema=True).na.drop()
display(df_stream)

# %%
#### Apply the model
from pyspark.sql.functions import current_timestamp

predictions = model.transform(df_stream).withColumn("timestamp", current_timestamp())
display(predictions.select("text", "prediction", "timestamp"))

# %%
#### Save predictions to persistent storage
predictions.select("text", "prediction", "timestamp") \
    .write.mode("append").option("header", True).csv("/FileStore/tables/stream_results")

# %%
#### Query predictions with Spark SQL
predictions.createOrReplaceTempView("stream_results")
spark.sql("SELECT prediction, COUNT(*) as total FROM stream_results GROUP BY prediction").show()
