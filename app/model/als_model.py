"""
ALS model loader — Spark + ALSModel singletons for restaurant recommendations.

Singletons are initialised once at startup via load_resources() and shared
across all requests — avoids the multi-second cost of re-creating a SparkSession.
"""

import logging
import os
import re
from typing import Dict, Optional

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType

logger = logging.getLogger(__name__)

# ── Path resolution (all overrideable via environment variables) ───────────────
_REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH       = os.getenv('MODEL_PATH',       os.path.join(_REPO_ROOT, 'model', 'saved_als_model'))
RESTAURANTS_CSV  = os.getenv('RESTAURANTS_CSV',  os.path.join(_REPO_ROOT, 'data',  'colombo_restaurants.csv'))

# ── Module-level singletons ────────────────────────────────────────────────────
_spark:               Optional[SparkSession] = None
_model:               Optional[ALSModel]     = None
_restaurants_lookup:  Dict[int, dict]        = {}   # restaurantId → metadata dict


# ── Metadata extraction helpers ───────────────────────────────────────────────

# Map Google Places type tokens → human-readable cuisine labels.
# Keys are checked in order against the comma-separated 'types' column.
_CUISINE_MAP: Dict[str, str] = {
    'indian_restaurant':        'Indian',
    'chinese_restaurant':       'Chinese',
    'italian_restaurant':       'Italian',
    'japanese_restaurant':      'Japanese',
    'thai_restaurant':          'Thai',
    'seafood_restaurant':       'Seafood',
    'fast_food_restaurant':     'Fast Food',
    'pizza_restaurant':         'Pizza',
    'hamburger_restaurant':     'Burgers',
    'vegetarian_restaurant':    'Vegetarian',
    'steak_house':              'Steakhouse',
    'sushi_restaurant':         'Sushi',
    'korean_restaurant':        'Korean',
    'vietnamese_restaurant':    'Vietnamese',
    'middle_eastern_restaurant':'Middle Eastern',
    'mexican_restaurant':       'Mexican',
    'turkish_restaurant':       'Turkish',
    'lebanese_restaurant':      'Lebanese',
    'bakery':                   'Bakery',
    'cafe':                     'Café',
    'bar':                      'Bar & Grill',
    'meal_takeaway':            'Takeaway',
    'meal_delivery':            'Delivery',
    'buffet_restaurant':        'Buffet',
    'breakfast_restaurant':     'Breakfast',
    'brunch_restaurant':        'Brunch',
    'food_court':               'Food Court',
    'ice_cream_shop':           'Desserts',
    'sandwich_shop':            'Sandwiches',
    'coffee_shop':              'Coffee',
}

_PRICE_LABELS: Dict[int, str] = {
    0: 'Budget',
    1: 'Inexpensive',
    2: 'Moderate',
    3: 'Expensive',
    4: 'Very Expensive',
}

# Named Colombo areas to recognise in address strings
_NAMED_AREAS = [
    'Nugegoda', 'Dehiwala', 'Mount Lavinia', 'Borella', 'Pettah',
    'Maradana', 'Wellawatte', 'Bambalapitiya', 'Kollupitiya',
    'Cinnamon Gardens', 'Fort',
]


def extract_cuisine(types_str: Optional[str]) -> str:
    """Return the most specific cuisine label found in a comma-separated Places types string."""
    if not types_str:
        return 'Restaurant'
    tokens = [t.strip().lower() for t in types_str.split(',')]
    for token in tokens:
        if token in _CUISINE_MAP:
            return _CUISINE_MAP[token]
    return 'Restaurant'


def extract_location(address: Optional[str]) -> str:
    """
    Extract the Colombo district or named area from a full address string.
    e.g. '10 Galle Road, Colombo 03, Sri Lanka' → 'Colombo 03'
    """
    if not address:
        return 'Colombo'
    # Prefer "Colombo NN" pattern
    match = re.search(r'Colombo\s+(\d+)', address, re.IGNORECASE)
    if match:
        return f'Colombo {int(match.group(1)):02d}'
    # Fall back to known named areas
    for area in _NAMED_AREAS:
        if area.lower() in address.lower():
            return area
    return 'Colombo'


def price_label(price_level) -> str:
    """Convert a numeric Google price_level (0–4) to a human-readable string."""
    try:
        return _PRICE_LABELS.get(int(price_level), 'Unknown')
    except (TypeError, ValueError):
        return 'Unknown'


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

    # Add derived columns used by analytics and response enrichment
    restaurants_df = (
        raw_df
        .withColumn('cuisine',     F.udf(extract_cuisine,  StringType())('types'))
        .withColumn('location',    F.udf(extract_location, StringType())('address'))
        .withColumn('price_range', F.udf(price_label,      StringType())('price_level'))
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
