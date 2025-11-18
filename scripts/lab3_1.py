#!/bin/python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

try:
    client = SparkSession.builder.appName("classifier").getOrCreate()
    df_iris = client.read.csv(
        path="hdfs://localhost:9000/data/iris.csv",
        header=True,
        inferSchema=True,
    )

    df_iris.printSchema()
    df_iris.show(3)

    assembler = VectorAssembler(
        inputCols=["X1", "X2", "X3", "X4"], outputCol="features"
    )
    indexer = StringIndexer(inputCol="Y", outputCol="label")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

    train_data, test_data = df_iris.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label")

    pipeline_lr = Pipeline(stages=[assembler, indexer, scaler, lr])
    model_lr = pipeline_lr.fit(train_data)
    
    predictions_lr = model_lr.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy_lr = evaluator.evaluate(predictions_lr, {evaluator.metricName: "accuracy"})

    print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")
finally:
    if client:
        client.stop()
