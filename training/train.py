"""
train.py — Entry point for the one-time ALS model training pipeline.

Orchestrates the training/ package modules in sequence:
    1. Load restaurant data              (training.data_loader)
    2. Generate synthetic ratings        (training.rating_generator)
    3. Train ALS model + evaluate RMSE   (training.model_trainer)
    4. Save serving artefacts            (training.artefact_builder)

Usage
-----
    python train.py

Prerequisites
-------------
Run the colombo-restaurant-pipeline first to produce:
    data/colombo_restaurants.csv

Outputs
-------
    model/saved_als_model/       — trained PySpark ALSModel
    model/top_restaurants.json   — top restaurants JSON (for /top-restaurants)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')

from app.config import (
    ALS_RANK, ALS_ITER, ALS_REG,
    TRAIN_SEED, TRAIN_SPLIT,
    N_USERS, RATINGS_PER_USER,
    MIN_REVIEWS_THRESHOLD, TOP_RESTAURANTS_LIMIT,
    TOP_RESTAURANTS_PATH,
)
from training.spark_session import build_training_spark
from training.data_loader import load_restaurants
from training.rating_generator import generate_ratings, ratings_to_spark_df
from training.model_trainer import train_als, save_model
from training.artefact_builder import build_top_restaurants_json

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
RESTAURANTS_CSV  = os.getenv('RESTAURANTS_CSV',  os.path.join(_SCRIPT_DIR, 'data',  'colombo_restaurants.csv'))
MODEL_OUTPUT_DIR = os.getenv('MODEL_PATH',        os.path.join(_SCRIPT_DIR, 'model', 'saved_als_model'))


def main() -> None:
    if not os.path.exists(RESTAURANTS_CSV):
        raise FileNotFoundError(
            f'\n❌  Restaurant CSV not found at: {RESTAURANTS_CSV}\n'
            '    Run the colombo-restaurant-pipeline first:\n'
            '        cd ../colombo-restaurant-pipeline && python pipeline.py\n'
            '    Then copy (or symlink) the output to data/colombo_restaurants.csv here.'
        )

    # 1. Spark session
    spark = build_training_spark()
    logger.info('SparkSession started.')

    # 2. Load restaurant data
    restaurants_df = load_restaurants(spark, RESTAURANTS_CSV)

    id_rating_rows = restaurants_df.select('restaurant_id', 'rating').collect()
    restaurant_ids = [r['restaurant_id'] for r in id_rating_rows]
    base_ratings   = {r['restaurant_id']: float(r['rating'] or 3.0) for r in id_rating_rows}

    # 3. Generate synthetic ratings
    records    = generate_ratings(restaurant_ids, base_ratings, N_USERS, RATINGS_PER_USER, TRAIN_SEED)
    ratings_df = ratings_to_spark_df(spark, records)

    # 4. Train ALS model
    model, rmse = train_als(
        ratings_df,
        train_split=TRAIN_SPLIT,
        seed=TRAIN_SEED,
        rank=ALS_RANK,
        max_iter=ALS_ITER,
        reg_param=ALS_REG,
    )
    print(f'\n✅  Test RMSE: {rmse:.4f}')

    save_model(model, MODEL_OUTPUT_DIR)

    # 5. Build serving artefacts
    build_top_restaurants_json(
        restaurants_df,
        output_path=TOP_RESTAURANTS_PATH,
        min_reviews=MIN_REVIEWS_THRESHOLD,
        limit=TOP_RESTAURANTS_LIMIT,
    )

    spark.stop()
    print(f'\n🎉  Training complete')
    print(f'    Model  → {MODEL_OUTPUT_DIR}')
    print(f'    Top    → {TOP_RESTAURANTS_PATH}\n')


if __name__ == '__main__':
    main()
