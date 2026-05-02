"""
FastAPI application entry point for the Colombo Restaurant Recommendation API.

Endpoints
---------
GET /                       health check
GET /recommend/{user_id}    top N restaurant recommendations (ALS + cold-start fallback)
GET /top-restaurants        most popular / highest-rated restaurants (static JSON)
GET /analytics/trending     Spark SQL analytics: top cuisines + busiest Colombo areas
"""

import logging
from pathlib import Path

# Load .env from the repo root before anything else reads os.getenv()
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.model.als_model import load_resources
from app.schemas.response import (
    RecommendationResponse,
    RestaurantListResponse,
    TopRestaurantsResponse,
    TrendingAnalyticsResponse,
)
from app.services.recommendation_service import (
    fetch_recommendations,
    fetch_restaurants,
    fetch_top_restaurants,
    fetch_trending_analytics,
)
from app.config import CORS_ORIGINS

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(name)s  %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Spark + ALS model once at startup; release on shutdown."""
    logger.info('⏳  Loading ALS model and restaurant data …')
    load_resources()
    logger.info('✅  Resources loaded — API is ready.')
    yield
    logger.info('🛑  Shutting down.')


app = FastAPI(
    title='Colombo Restaurant Recommendation API',
    description='ALS-powered restaurant recommendations with Colombo analytics.',
    version='2.0.0',
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/', tags=['Health'])
def health_check():
    return {'status': 'ok', 'service': 'Colombo Restaurant Recommendation API'}


@app.get('/restaurants', response_model=RestaurantListResponse, tags=['Restaurants'])
def get_restaurants(
    cuisine:     Optional[str]   = Query(None),
    district:    Optional[str]   = Query(None),
    price_range: Optional[str]   = Query(None),
    min_rating:  Optional[float] = Query(None, ge=0, le=5),
    sort_by:     str             = Query(default='rating', regex='^(rating|popularity|price|reviews)$'),
    limit:       int             = Query(default=50, ge=1, le=200),
):
    """
    Browse all restaurants with optional filters.

    Queries the 'restaurants' Spark temp view registered from colombo_restaurants.csv.
    """
    try:
        restaurants, total = fetch_restaurants(
            cuisine=cuisine,
            district=district,
            price_range=price_range,
            min_rating=min_rating,
            sort_by=sort_by,
            limit=limit,
        )
        return RestaurantListResponse(restaurants=restaurants, total=total)
    except Exception as exc:
        logger.exception('Error fetching restaurants')
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/recommend/{user_id}', response_model=RecommendationResponse, tags=['Recommendations'])
def get_recommendations(
    user_id: int,
    top_n:   int = Query(default=10, ge=1, le=50, description='Number of recommendations to return'),
):
    """
    Return the top N restaurant recommendations for *user_id*.

    If the user was not seen during model training (cold start), the response
    will contain the top-rated restaurants instead and `is_cold_start` will be
    `true` — the HTTP status is always 200.
    """
    try:
        recs, is_cold_start = fetch_recommendations(user_id, top_n)
        return RecommendationResponse(
            userId=user_id,
            recommendations=recs,
            is_cold_start=is_cold_start,
        )
    except Exception as exc:
        logger.exception('Error generating recommendations for user %d', user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/top-restaurants', response_model=TopRestaurantsResponse, tags=['Restaurants'])
def get_top_restaurants(
    limit: int = Query(default=20, ge=1, le=100, description='How many restaurants to return'),
):
    """Return the most popular restaurants pre-computed from the training dataset."""
    try:
        restaurants = fetch_top_restaurants(limit)
        return TopRestaurantsResponse(restaurants=restaurants)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Error fetching top restaurants')
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/analytics/trending', response_model=TrendingAnalyticsResponse, tags=['Analytics'])
def get_trending_analytics():
    """
    Spark SQL analytics over colombo_restaurants.csv:
    - Top 5 cuisines by restaurant count and average rating
    - Top 5 busiest Colombo areas by average synthetic busyness score
    """
    try:
        data = fetch_trending_analytics()
        return TrendingAnalyticsResponse(**data)
    except Exception as exc:
        logger.exception('Error computing trending analytics')
        raise HTTPException(status_code=500, detail=str(exc)) from exc
