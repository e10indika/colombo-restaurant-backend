"""
train_and_save.py — one-time training script for the Colombo Restaurant ALS model.

Prerequisites
-------------
Run the colombo-restaurant-pipeline to produce:
    data/colombo_restaurants.csv

Usage
-----
    python train_and_save.py

Outputs
-------
    model/saved_als_model/   — trained PySpark ALSModel
    model/top_restaurants.json — top 50 restaurants by rating (for /top-restaurants endpoint)
"""

import json
import logging
import math
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
RESTAURANTS_CSV      = os.getenv('RESTAURANTS_CSV',  os.path.join(_SCRIPT_DIR, 'data',  'colombo_restaurants.csv'))
MODEL_OUTPUT_DIR     = os.getenv('MODEL_PATH',        os.path.join(_SCRIPT_DIR, 'model', 'saved_als_model'))
TOP_RESTAURANTS_PATH = os.getenv('TOP_RESTAURANTS_PATH', os.path.join(_SCRIPT_DIR, 'model', 'top_restaurants.json'))

# ── ALS hyperparameters ────────────────────────────────────────────────────────
ALS_RANK    = 10
ALS_ITER    = 10
ALS_REG     = 0.1

# ── Synthetic data generation config ─────────────────────────────────────────
N_USERS            = 200
RATINGS_PER_USER   = (20, 50)   # uniform random in this range
random.seed(42)


# ── Metadata helpers (mirrors als_model.py) ───────────────────────────────────

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
    'bakery':                   'Bakery',
    'cafe':                     'Café',
    'bar':                      'Bar & Grill',
    'buffet_restaurant':        'Buffet',
    'food_court':               'Food Court',
    'ice_cream_shop':           'Desserts',
    'coffee_shop':              'Coffee',
}

_PRICE_LABELS: Dict[int, str] = {
    0: 'Budget',
    1: 'Inexpensive',
    2: 'Moderate',
    3: 'Expensive',
    4: 'Very Expensive',
}

_NAMED_AREAS = [
    'Nugegoda', 'Dehiwala', 'Mount Lavinia', 'Borella', 'Pettah',
    'Maradana', 'Wellawatte', 'Bambalapitiya', 'Kollupitiya',
    'Cinnamon Gardens', 'Fort',
]


def _extract_cuisine(types_str: Optional[str]) -> str:
    if not types_str:
        return 'Restaurant'
    tokens = [t.strip().lower() for t in types_str.split(',')]
    for token in tokens:
        if token in _CUISINE_MAP:
            return _CUISINE_MAP[token]
    return 'Restaurant'


def _extract_location(address: Optional[str]) -> str:
    if not address:
        return 'Colombo'
    match = re.search(r'Colombo\s+(\d+)', address or '', re.IGNORECASE)
    if match:
        return f'Colombo {int(match.group(1)):02d}'
    for area in _NAMED_AREAS:
        if area.lower() in (address or '').lower():
            return area
    return 'Colombo'


def _price_label(price_level) -> str:
    try:
        return _PRICE_LABELS.get(int(price_level), 'Unknown')
    except (TypeError, ValueError):
        return 'Unknown'


# ── Main training routine ─────────────────────────────────────────────────────

def main():
    # ── Validate input file ────────────────────────────────────────────────────
    if not os.path.exists(RESTAURANTS_CSV):
        raise FileNotFoundError(
            f'\n❌  Restaurant CSV not found at: {RESTAURANTS_CSV}\n'
            '    Run the colombo-restaurant-pipeline first:\n'
            '        cd ../colombo-restaurant-pipeline && python pipeline.py\n'
            '    Then copy (or symlink) the output to data/colombo_restaurants.csv here.'
        )

    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import FloatType, IntegerType

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
    logger.info('SparkSession started.')

    # ── Load and sort CSV (ordering must match als_model.py) ──────────────────
    raw_df = (
        spark.read
        .option('header', 'true')
        .csv(RESTAURANTS_CSV)
        .withColumn('rating',        F.col('rating')       .cast(FloatType()))
        .withColumn('total_ratings', F.col('total_ratings').cast(IntegerType()))
        .withColumn('price_level',   F.col('price_level')  .cast(IntegerType()))
        .dropDuplicates(['place_id'])
        .orderBy('place_id')
    )

    # Assign deterministic integer restaurant_id (1-based, sorted by place_id)
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, lit

    restaurants_df = raw_df.withColumn(
        'restaurant_id',
        row_number().over(Window.orderBy('place_id')).cast(IntegerType()),
    )

    n_restaurants = restaurants_df.count()
    logger.info('Loaded %d unique restaurants.', n_restaurants)

    # Collect the data needed for synthetic rating generation
    id_rating_rows = restaurants_df.select('restaurant_id', 'rating').collect()
    restaurant_ids   = [row['restaurant_id'] for row in id_rating_rows]
    base_ratings     = {row['restaurant_id']: float(row['rating'] or 3.0) for row in id_rating_rows}

    # ── Generate synthetic user-restaurant ratings ────────────────────────────
    logger.info('Generating synthetic ratings for %d users …', N_USERS)
    rating_records = []
    import math

    for user_id in range(1, N_USERS + 1):
        n_ratings    = random.randint(*RATINGS_PER_USER)
        sampled_rids = random.sample(restaurant_ids, min(n_ratings, len(restaurant_ids)))
        user_bias    = random.gauss(0, 0.3)   # each user has a systematic over/under-rater bias

        for rid in sampled_rids:
            base    = base_ratings[rid]
            noise   = random.gauss(0, 0.5)
            raw     = base + user_bias + noise
            clamped = max(0.5, min(5.0, raw))
            rating_records.append((user_id, rid, round(clamped, 1)))

    logger.info('Total synthetic ratings: %d', len(rating_records))

    from pyspark.sql.types import StructType, StructField

    ratings_df = spark.createDataFrame(
        rating_records,
        schema=['userId', 'restaurantId', 'rating'],
    )

    # ── Train / test split ────────────────────────────────────────────────────
    train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # ── ALS model ─────────────────────────────────────────────────────────────
    als = ALS(
        rank=ALS_RANK,
        maxIter=ALS_ITER,
        regParam=ALS_REG,
        userCol='userId',
        itemCol='restaurantId',
        ratingCol='rating',
        coldStartStrategy='drop',
        nonnegative=True,
    )

    logger.info('Training ALS (rank=%d, maxIter=%d, regParam=%s) …', ALS_RANK, ALS_ITER, ALS_REG)
    model = als.fit(train_df)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    predictions = model.transform(test_df)
    evaluator   = RegressionEvaluator(
        metricName='rmse',
        labelCol='rating',
        predictionCol='prediction',
    )
    rmse = evaluator.evaluate(predictions)
    logger.info('Test RMSE: %.4f', rmse)
    print(f'\n✅  Test RMSE: {rmse:.4f}')

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUTPUT_DIR), exist_ok=True)
    model.write().overwrite().save(MODEL_OUTPUT_DIR)
    logger.info('ALS model saved to %s', MODEL_OUTPUT_DIR)

    # ── Build top_restaurants.json ─────────────────────────────────────────────
    # Top 50 restaurants by Google rating, requiring at least 10 ratings
    top_rows = (
        restaurants_df
        .filter(F.col('total_ratings') >= 10)
        .orderBy(F.col('rating').desc(), F.col('total_ratings').desc())
        .limit(50)
        .select(
            'restaurant_id', 'place_id', 'name', 'address',
            'rating', 'total_ratings', 'price_level',
            'types', 'busyness_score',
        )
        .collect()
    )

    top_list = [
        {
            'restaurantId':  int(row['restaurant_id']),
            'name':          row['name']         or 'Unknown',
            'cuisine':       _extract_cuisine(row['types']),
            'location':      _extract_location(row['address']),
            'price_range':   _price_label(row['price_level']),
            'rating':        float(row['rating'] or 0.0),
            'total_ratings': int(row['total_ratings'] or 0),
            'busyness_score': float(row['busyness_score'] or 0.0) if row['busyness_score'] else 0.0,
        }
        for row in top_rows
    ]

    os.makedirs(os.path.dirname(TOP_RESTAURANTS_PATH), exist_ok=True)
    with open(TOP_RESTAURANTS_PATH, 'w') as f:
        json.dump(top_list, f, indent=2)
    logger.info('Top restaurants saved to %s (%d entries)', TOP_RESTAURANTS_PATH, len(top_list))

    spark.stop()
    print(f'\n🎉  Training complete — model saved to {MODEL_OUTPUT_DIR}')
    print(f'    top_restaurants.json saved to {TOP_RESTAURANTS_PATH}\n')


if __name__ == '__main__':
    main()
