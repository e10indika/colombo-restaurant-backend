"""
training/rating_generator.py — Generate a synthetic user-restaurant rating matrix.

Each synthetic user samples a random subset of restaurants and rates them using:
    rating = google_rating + user_bias + noise,  clamped to [0.5, 5.0]

This lets us train an ALS collaborative-filtering model even though we have no
real user interaction data.  The Google rating anchors the synthetic signal so
the model learns meaningful latent representations of restaurant quality.
"""

import csv
import logging
import os
import random
import tempfile
from typing import List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType

logger = logging.getLogger(__name__)

RatingRecord = Tuple[int, int, float]   # (userId, restaurantId, rating)


def generate_ratings(
    restaurant_ids: List[int],
    base_ratings: dict,
    n_users: int,
    ratings_per_user: Tuple[int, int],
    seed: int,
) -> List[RatingRecord]:
    """
    Build the in-memory list of (userId, restaurantId, rating) triples.

    Args:
        restaurant_ids:   List of integer restaurant IDs.
        base_ratings:     Map of restaurant_id → google rating (float).
        n_users:          Number of synthetic users to generate.
        ratings_per_user: (min, max) number of restaurants each user rates.
        seed:             Random seed for reproducibility.

    Returns:
        List of (userId, restaurantId, rating) tuples.
    """
    random.seed(seed)
    records: List[RatingRecord] = []

    for user_id in range(1, n_users + 1):
        n_ratings    = random.randint(*ratings_per_user)
        sampled      = random.sample(restaurant_ids, min(n_ratings, len(restaurant_ids)))
        user_bias    = random.gauss(0, 0.3)   # systematic over/under-rater bias

        for rid in sampled:
            raw     = base_ratings[rid] + user_bias + random.gauss(0, 0.5)
            clamped = max(0.5, min(5.0, raw))
            records.append((user_id, rid, round(clamped, 1)))

    logger.info('Generated %d synthetic ratings for %d users', len(records), n_users)
    return records


def ratings_to_spark_df(
    spark: SparkSession,
    records: List[RatingRecord],
) -> DataFrame:
    """
    Convert the in-memory rating records to a Spark DataFrame.

    We write to a temp CSV and read it back to avoid the Python→JVM pickle
    path that causes RecursionError on Python 3.12+ with PySpark 3.5.x.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv', prefix='ratings_')
    try:
        with os.fdopen(tmp_fd, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['userId', 'restaurantId', 'rating'])
            writer.writerows(records)

        df = (
            spark.read
            .option('header', 'true')
            .csv(tmp_path)
            .withColumn('userId',       F.col('userId')      .cast(IntegerType()))
            .withColumn('restaurantId', F.col('restaurantId').cast(IntegerType()))
            .withColumn('rating',       F.col('rating')      .cast(FloatType()))
        )
        df.cache()
        logger.info('Ratings DataFrame cached: %d rows', df.count())
        return df
    finally:
        os.unlink(tmp_path)
