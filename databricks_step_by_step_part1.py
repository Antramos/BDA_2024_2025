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
# Databricks Part 1 – Training and Saving the Model

# %%
#### Load fake and real datasets
df_fake = spark.read.csv("/FileStore/tables/fake.csv", header=True, inferSchema=True)
df_real = spark.read.csv("/FileStore/tables/real.csv", header=True, inferSchema=True)

# %%
#### Label and combine the datasets
from pyspark.sql.functions import lit

df_fake = df_fake.withColumn("label", lit(0))
df_real = df_real.withColumn("label", lit(1))
df = df_fake.unionByName(df_real).select("text", "label").na.drop()
display(df)

# %%
df.groupBy("label").count().show()

# %%
#### Preprocess text (remove symbols, lowercase)
from pyspark.sql.functions import lower, regexp_replace

df = df.withColumn("text", lower(regexp_replace("text", "[^a-zA-Z\s]", "")))
display(df)

# %%
#### Train/test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# %%
#### Define pipeline with TF-IDF and Random Forest
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
tf = HashingTF(inputCol="filtered", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")

# %%
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10, maxDepth=4)

pipeline_rf = Pipeline(stages=[tokenizer, remover, tf, idf, rf])

# %%
#### Naive Bayes baseline
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(labelCol="label", featuresCol="features", modelType="multinomial")
pipeline_nb = Pipeline(stages=[tokenizer, remover, tf, idf, nb])

# %%
#### Train both models
model_rf = pipeline_rf.fit(train_data)
model_nb = pipeline_nb.fit(train_data)

# %%
#### Evaluate both models
from pyspark.ml.evaluation import BinaryClassificationEvaluator

pred_rf = model_rf.transform(test_data)
pred_nb = model_nb.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

auc_rf = evaluator.evaluate(pred_rf)
acc_rf = pred_rf.filter("label = prediction").count() / pred_rf.count()

auc_nb = evaluator.evaluate(pred_nb)
acc_nb = pred_nb.filter("label = prediction").count() / pred_nb.count()

print("🌲 Random Forest:")
print(f"AUC: {auc_rf:.4f}")
print(f"Accuracy: {acc_rf:.4f}")
print("\n🤖 Naive Bayes:")
print(f"AUC: {auc_nb:.4f}")
print(f"Accuracy: {acc_nb:.4f}")

# %%
#### Save best model (we can pick RF or NB and others that we can implement), analyze what's best and, until now, the best was Random Forest.
model_rf.write().overwrite().save("dbfs:/FileStore/models/fake_news_best_model")
print("** Random Forest model saved.")

# %%
dbutils.fs.ls("dbfs:/FileStore/models/fake_news_best_model/")
