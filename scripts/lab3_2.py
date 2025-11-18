#!/bin/python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

client = None

try:
    client = SparkSession.builder.appName("regress").getOrCreate()
    df_wine_red = client.read.csv(
        path="hdfs://localhost:9000/data/winequality-red.csv",
        header=True,
        inferSchema=True,
        sep=";",
        )
    df_wine_white = client.read.csv(
        path="hdfs://localhost:9000/data/winequality-white.csv",
        header=True,
        inferSchema=True,
        sep=";",
        )
    df = df_wine_red.union(df_wine_white)

    df.printSchema()
    df.show(3)
    
    assembler = VectorAssembler(
        inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        outputCol="features",
    )
    pca = PCA(k=6, inputCol="features", outputCol="pcaFeatures")
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    rf = RandomForestRegressor(
        featuresCol="pcaFeatures",
        labelCol="quality",
        numTrees=100,
        maxDepth=10,
        seed=42 
    )

    pipeline_reg = Pipeline(stages=[assembler, pca, rf])
    model_reg = pipeline_reg.fit(train_data)
    predictions_reg = model_reg.transform(test_data)

    evaluator_reg = RegressionEvaluator(labelCol="quality", predictionCol="prediction")
    rmse = evaluator_reg.evaluate(predictions_reg, {evaluator_reg.metricName:"rmse"})
    mae = evaluator_reg.evaluate(predictions_reg, {evaluator_reg.metricName:"mae"})
    
    print(f"Linear Regression RMSE: {rmse:.2f}, MAE: {mae:.2f}")
finally:
    if client:
        client.stop()