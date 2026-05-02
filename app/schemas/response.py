"""
Pydantic response models — defines the exact JSON shape for every endpoint.
Keeping schemas here decouples serialisation from business logic.
"""

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status:  str = Field(..., example='ok')
    message: str = Field(..., example='Colombo Restaurant Recommendation API is running.')


# ── /recommend/{user_id} ──────────────────────────────────────────────────────

class RestaurantRecommendation(BaseModel):
    restaurantId:     int   = Field(..., example=42)
    name:             str   = Field(..., example='The Lagoon — Cinnamon Grand')
    cuisine:          str   = Field(..., example='Seafood')
    location:         str   = Field(..., example='Colombo 03')
    price_range:      str   = Field(..., example='Expensive')
    predicted_rating: float = Field(..., example=4.72)
    busyness_score:   float = Field(..., example=83.4)


class RecommendationResponse(BaseModel):
    userId:          int                         = Field(..., example=1)
    is_cold_start:   bool                        = Field(
        False,
        description='True when the user was not in the training data; '
                    'results are top-rated restaurants instead of personalised picks.',
        example=False,
    )
    recommendations: List[RestaurantRecommendation]


# ── /top-restaurants ─────────────────────────────────────────────────────────

class TopRestaurant(BaseModel):
    restaurantId:  int   = Field(..., example=7)
    name:          str   = Field(..., example='Ministry of Crab')
    cuisine:       str   = Field(..., example='Seafood')
    location:      str   = Field(..., example='Colombo 01')
    price_range:   str   = Field(..., example='Very Expensive')
    total_ratings: int   = Field(..., example=4821)
    rating:        float = Field(..., example=4.6)
    busyness_score: float = Field(..., example=91.2)


class TopRestaurantsResponse(BaseModel):
    restaurants: List[TopRestaurant]


# ── /analytics/trending ───────────────────────────────────────────────────────

class TrendingCuisine(BaseModel):
    cuisine:        str   = Field(..., example='Seafood')
    restaurant_count: int = Field(..., example=34)
    avg_rating:     float = Field(..., example=4.3)


class TrendingArea(BaseModel):
    location:          str   = Field(..., example='Colombo 03')
    restaurant_count:  int   = Field(..., example=58)
    avg_busyness_score: float = Field(..., example=76.5)
    avg_rating:        float = Field(..., example=4.1)


class TrendingAnalyticsResponse(BaseModel):
    top_cuisines:  List[TrendingCuisine]
    busiest_areas: List[TrendingArea]
