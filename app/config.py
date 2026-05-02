"""
Application configuration — all environment-driven settings in one place.
Import from here rather than calling os.getenv() scattered across modules.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── File paths ────────────────────────────────────────────────────────────────
RESTAURANTS_CSV      = os.getenv('RESTAURANTS_CSV',
    str(_REPO_ROOT / 'data' / 'colombo_restaurants.csv'))
MODEL_PATH           = os.getenv('MODEL_PATH',
    str(_REPO_ROOT / 'model' / 'saved_als_model'))
TOP_RESTAURANTS_PATH = os.getenv('TOP_RESTAURANTS_PATH',
    str(_REPO_ROOT / 'model' / 'top_restaurants.json'))

# ── API / server ──────────────────────────────────────────────────────────────
CORS_ORIGINS: list = [
    o.strip()
    for o in os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    if o.strip()
]

# ── ALS hyper-parameters ──────────────────────────────────────────────────────
ALS_RANK   = 10
ALS_ITER   = 10
ALS_REG    = 0.1
TRAIN_SEED = 42
TRAIN_SPLIT = [0.8, 0.2]

# ── Synthetic data generation ─────────────────────────────────────────────────
N_USERS          = 200
RATINGS_PER_USER = (20, 50)

# ── Top-restaurants thresholds ────────────────────────────────────────────────
MIN_REVIEWS_THRESHOLD = 10
TOP_RESTAURANTS_LIMIT = 50
