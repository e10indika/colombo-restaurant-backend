"""
training/artefact_builder.py — Build and persist serving artefacts from the
trained model's restaurant DataFrame.

Currently produces:
    top_restaurants.json — top N restaurants by Google rating, used by the
                           /top-restaurants endpoint to avoid a Spark query
                           on every request.
"""

import json
import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from app.utils.metadata import extract_cuisine, extract_location, price_label

logger = logging.getLogger(__name__)


def build_top_restaurants_json(
    restaurants_df: DataFrame,
    output_path: str,
    min_reviews: int,
    limit: int,
) -> None:
    """
    Select the top `limit` restaurants by Google rating (requiring at least
    `min_reviews` ratings), enrich with cuisine/location/price labels, and
    write to `output_path` as JSON.

    Args:
        restaurants_df: DataFrame with columns restaurant_id, place_id, name,
                        address, rating, total_ratings, price_level, types,
                        busyness_score.
        output_path:    Destination path for the JSON file.
        min_reviews:    Minimum total_ratings required to appear in the list.
        limit:          Maximum number of entries to include.
    """
    rows = (
        restaurants_df
        .filter(F.col('total_ratings') >= min_reviews)
        .orderBy(F.col('rating').desc(), F.col('total_ratings').desc())
        .limit(limit)
        .select(
            'restaurant_id', 'place_id', 'name', 'address',
            'lat', 'lng',
            'rating', 'total_ratings', 'price_level',
            'types', 'busyness_score',
        )
        .collect()
    )

    top_list = [
        {
            'restaurantId':   int(row['restaurant_id']),
            'name':           row['name']          or 'Unknown',
            'cuisine':        extract_cuisine(row['types']),
            'location':       extract_location(row['address']),
            'price_range':    price_label(row['price_level']),
            'rating':         float(row['rating']          or 0.0),
            'total_ratings':  int(row['total_ratings']     or 0),
            'busyness_score': float(row['busyness_score']  or 0.0) if row['busyness_score'] else 0.0,
            'lat':            float(row['lat'])  if row['lat']  else None,
            'lng':            float(row['lng'])  if row['lng']  else None,
        }
        for row in rows
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as fh:
        json.dump(top_list, fh, indent=2)

    logger.info('Top restaurants JSON saved to %s (%d entries)', output_path, len(top_list))
