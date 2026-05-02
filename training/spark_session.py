"""
training/spark_session.py — SparkSession factory for the training pipeline.

Kept separate from app/model/als_model.py (which manages the inference session)
so the training job can be run independently without importing the full FastAPI app.
"""

from pyspark.sql import SparkSession


def build_training_spark() -> SparkSession:
    """Create a local SparkSession optimised for training (not inference)."""
    spark = (
        SparkSession.builder
        .appName('ColomboRestaurantTrainer')
        .master('local[*]')
        .config('spark.driver.memory', '3g')
        .config('spark.sql.shuffle.partitions', '8')
        .config('spark.ui.enabled', 'false')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('WARN')
    return spark
