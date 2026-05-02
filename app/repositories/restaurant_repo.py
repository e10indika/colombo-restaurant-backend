"""
Restaurant repository — data access layer for the 'restaurants' Spark temp view.
All Spark SQL lives here; the service layer stays free of query strings.
"""
from typing import Any, Dict, List, Optional, Tuple
from pyspark.sql import SparkSession


_SORT_EXPR: Dict[str, str] = {
    'rating':     'rating DESC NULLS LAST',
    'popularity': 'busyness_score DESC NULLS LAST',
    'reviews':    'total_ratings DESC NULLS LAST',
    'price': (
        "CASE price_range "
        "WHEN 'Free'      THEN 0 "
        "WHEN 'Budget'    THEN 1 "
        "WHEN 'Moderate'  THEN 2 "
        "WHEN 'Expensive' THEN 3 "
        "WHEN 'Luxury'    THEN 4 "
        "ELSE 5 END ASC"
    ),
}


def _build_where(
    cuisine:     Optional[str],
    district:    Optional[str],
    price_range: Optional[str],
    min_rating:  Optional[float],
) -> str:
    conditions = []
    if cuisine:
        conditions.append(f"cuisine = '{cuisine.replace(chr(39), chr(39)*2)}'")
    if district:
        conditions.append(f"location = '{district.replace(chr(39), chr(39)*2)}'")
    if price_range:
        conditions.append(f"price_range = '{price_range.replace(chr(39), chr(39)*2)}'")
    if min_rating is not None:
        conditions.append(f"rating >= {float(min_rating)}")
    return ('WHERE ' + ' AND '.join(conditions)) if conditions else ''


def query_restaurants(
    spark:       SparkSession,
    cuisine:     Optional[str]   = None,
    district:    Optional[str]   = None,
    price_range: Optional[str]   = None,
    min_rating:  Optional[float] = None,
    sort_by:     str             = 'rating',
    limit:       int             = 50,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Query the 'restaurants' Spark temp view with optional filters.

    Returns
    -------
    (rows_as_dicts, total_matching_count)
    """
    where  = _build_where(cuisine, district, price_range, min_rating)
    order  = _SORT_EXPR.get(sort_by, _SORT_EXPR['rating'])

    rows = spark.sql(f"""
        SELECT
            place_id                                                    AS restaurantId,
            name,
            cuisine,
            location,
            price_range,
            ROUND(COALESCE(CAST(rating        AS DOUBLE), 0.0), 2)     AS rating,
            CAST (COALESCE(CAST(total_ratings AS INT),    0)   AS INT)  AS total_ratings,
            ROUND(COALESCE(CAST(busyness_score AS DOUBLE), 0.0), 1)    AS busyness_score
        FROM restaurants
        {where}
        ORDER BY {order}
        LIMIT {int(limit)}
    """).collect()

    total = int(spark.sql(
        f"SELECT COUNT(*) AS cnt FROM restaurants {where}"
    ).collect()[0]['cnt'])

    return [
        {
            'restaurantId':   str(r['restaurantId']),
            'name':           r['name']        or 'Unknown',
            'cuisine':        r['cuisine']     or 'Restaurant',
            'location':       r['location']    or 'Colombo',
            'price_range':    r['price_range'] or 'Unknown',
            'rating':         float(r['rating']),
            'total_ratings':  int(r['total_ratings']),
            'busyness_score': float(r['busyness_score']),
        }
        for r in rows
    ], total
