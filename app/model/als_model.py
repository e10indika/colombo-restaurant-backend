"""
ALS model loader — Spark + ALSModel singletons for restaurant recommendations.

Singletons are initialised once at startup via load_resources() and shared
across all requests — avoids the multi-second cost of re-creating a SparkSession.
"""

import logging
import os

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType

from app.utils.metadata import _CUISINE_MAP, extract_cuisine, extract_location, price_label
from app.config import MODEL_PATH, RESTAURANTS_CSV

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_spark:               SparkSession | None = None
_model:               ALSModel | None     = None
_restaurants_lookup:  dict                = {}   # restaurantId → metadata dict


# ── Spark setup ────────────────────────────────────────────────────────────────

def _init_spark() -> SparkSession:
    global _spark
    if _spark is None:
        logger.info('Initialising SparkSession …')
        _spark = (
            SparkSession.builder
            .appName('Colombo-Restaurant-API')
            .master('local[2]')
            .config('spark.driver.memory', '2g')
            .config('spark.sql.shuffle.partitions', '8')
            .config('spark.ui.enabled', 'false')
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel('WARN')
        logger.info('SparkSession ready.')
    return _spark


# ── Resource loading ───────────────────────────────────────────────────────────

def load_resources() -> None:
    """
    Load the ALS model and restaurant metadata.
    Called once during the FastAPI lifespan startup event.

    Side-effects
    ------------
    - Populates _model and _restaurants_lookup singletons.
    - Registers a Spark SQL temp view named 'restaurants' for analytics queries.
    """
    global _model, _restaurants_lookup

    spark = _init_spark()

    # ── ALS model ─────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'ALS model not found at {MODEL_PATH}. '
            'Run `python train_and_save.py` first.'
        )
    logger.info('Loading ALS model from %s …', MODEL_PATH)
    _model = ALSModel.load(MODEL_PATH)
    logger.info('ALS model loaded.')

    # ── Restaurant CSV ────────────────────────────────────────────────────────
    if not os.path.exists(RESTAURANTS_CSV):
        raise FileNotFoundError(
            f'Restaurant data not found at {RESTAURANTS_CSV}. '
            'Run the colombo-restaurant-pipeline first to generate the CSV.'
        )

    logger.info('Loading restaurant data from %s …', RESTAURANTS_CSV)

    raw_df = (
        spark.read
        .option('header', 'true')
        .csv(RESTAURANTS_CSV)
        # Cast numeric columns from inferred strings
        .withColumn('rating',         F.col('rating')        .cast(FloatType()))
        .withColumn('total_ratings',  F.col('total_ratings') .cast(IntegerType()))
        .withColumn('price_level',    F.col('price_level')   .cast(IntegerType()))
        .withColumn('busyness_score', F.col('busyness_score').cast(FloatType()))
        .dropDuplicates(['place_id'])
        # Assign a stable integer restaurant_id by sorting on place_id
        # (must match the ordering used in train_and_save.py)
        .orderBy('place_id')
    )

    # Add derived columns — pure Spark expressions, no UDFs (UDFs cause
    # RecursionError when Spark tries to pickle module-level globals).
    cuisine_col = (
        F.when(F.col('types').contains('indian_restaurant'),        'Indian')
         .when(F.col('types').contains('chinese_restaurant'),       'Chinese')
         .when(F.col('types').contains('italian_restaurant'),       'Italian')
         .when(F.col('types').contains('japanese_restaurant'),      'Japanese')
         .when(F.col('types').contains('thai_restaurant'),          'Thai')
         .when(F.col('types').contains('seafood_restaurant'),       'Seafood')
         .when(F.col('types').contains('fast_food_restaurant'),     'Fast Food')
         .when(F.col('types').contains('pizza_restaurant'),         'Pizza')
         .when(F.col('types').contains('hamburger_restaurant'),     'Burgers')
         .when(F.col('types').contains('vegetarian_restaurant'),    'Vegetarian')
         .when(F.col('types').contains('steak_house'),              'Steakhouse')
         .when(F.col('types').contains('sushi_restaurant'),         'Sushi')
         .when(F.col('types').contains('korean_restaurant'),        'Korean')
         .when(F.col('types').contains('vietnamese_restaurant'),    'Vietnamese')
         .when(F.col('types').contains('middle_eastern_restaurant'),'Middle Eastern')
         .when(F.col('types').contains('bakery'),                   'Bakery')
         .when(F.col('types').contains('cafe'),                     'Café')
         .when(F.col('types').contains('bar'),                      'Bar & Grill')
         .when(F.col('types').contains('ice_cream_shop'),           'Desserts')
         .otherwise('Restaurant')
    )

    location_col = F.when(
        F.regexp_extract(F.col('address'), r'(?i)Colombo\s+(\d+)', 1) != '',
        F.concat(F.lit('Colombo '), F.lpad(
            F.regexp_extract(F.col('address'), r'(?i)Colombo\s+(\d+)', 1), 2, '0'
        )),
    ).otherwise(F.lit('Colombo'))

    price_range_col = (
        F.when(F.col('price_level') == 0, 'Budget')
         .when(F.col('price_level') == 1, 'Inexpensive')
         .when(F.col('price_level') == 2, 'Moderate')
         .when(F.col('price_level') == 3, 'Expensive')
         .when(F.col('price_level') == 4, 'Very Expensive')
         .otherwise('Unknown')
    )

    restaurants_df = (
        raw_df
        .withColumn('cuisine',     cuisine_col)
        .withColumn('location',    location_col)
        .withColumn('price_range', price_range_col)
    )

    # Register as Spark SQL temp view so the analytics service can run SQL queries
    restaurants_df.createOrReplaceTempView('restaurants')
    logger.info('Spark SQL temp view "restaurants" registered.')

    # Build Python dict for O(1) per-request lookup (avoids a Spark job per request)
    rows = restaurants_df.select(
        'place_id', 'name', 'cuisine', 'location', 'price_range',
        'rating', 'total_ratings', 'busyness_score',
    ).collect()

    # restaurant_id = 1-based row position after ordering by place_id
    # Must match the sequential IDs assigned during training
    _restaurants_lookup = {
        idx + 1: {
            'place_id':      row['place_id'],
            'name':          row['name']          or 'Unknown',
            'cuisine':       row['cuisine']        or 'Restaurant',
            'location':      row['location']       or 'Colombo',
            'price_range':   row['price_range']    or 'Unknown',
            'rating':        float(row['rating'] or 0.0),
            'total_ratings': int(row['total_ratings'] or 0),
            'busyness_score': float(row['busyness_score'] or 0.0),
        }
        for idx, row in enumerate(rows)
    }
    logger.info('Restaurant lookup built: %d entries.', len(_restaurants_lookup))


# ── Accessors ──────────────────────────────────────────────────────────────────

def get_model() -> ALSModel:
    if _model is None:
        raise RuntimeError('ALS model not loaded. Call load_resources() first.')
    return _model


def get_spark() -> SparkSession:
    if _spark is None:
        raise RuntimeError('SparkSession not initialised. Call load_resources() first.')
    return _spark


def get_restaurants_lookup() -> Dict[int, dict]:
    return _restaurants_lookup
