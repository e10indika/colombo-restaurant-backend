"""
training/data_loader.py — Load the restaurant CSV and assign stable integer IDs.

The integer restaurant_id is derived deterministically from place_id sort order
so it matches the index used by als_model.py at inference time.
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def load_restaurants(spark: SparkSession, csv_path: str) -> DataFrame:
    """
    Read the restaurant CSV, cast columns, deduplicate, and assign a
    1-based integer restaurant_id ordered by place_id.

    Returns a DataFrame with all original columns plus `restaurant_id`.
    """
    raw_df = (
        spark.read
        .option('header', 'true')
        .csv(csv_path)
        .withColumn('rating',        F.col('rating')       .cast(FloatType()))
        .withColumn('total_ratings', F.col('total_ratings').cast(IntegerType()))
        .withColumn('price_level',   F.col('price_level')  .cast(IntegerType()))
        .dropDuplicates(['place_id'])
        .orderBy('place_id')
    )

    restaurants_df = raw_df.withColumn(
        'restaurant_id',
        F.row_number().over(Window.orderBy('place_id')).cast(IntegerType()),
    )

    count = restaurants_df.count()
    logger.info('Loaded %d unique restaurants from %s', count, csv_path)
    return restaurants_df
