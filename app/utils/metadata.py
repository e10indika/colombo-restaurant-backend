"""
Domain metadata — cuisine mapping, price labels, location extraction.
Single source of truth shared by als_model.py and train_and_save.py.
"""
from typing import Dict, Optional
import re

_CUISINE_MAP: Dict[str, str] = {
    'indian_restaurant':         'Indian',
    'chinese_restaurant':        'Chinese',
    'italian_restaurant':        'Italian',
    'japanese_restaurant':       'Japanese',
    'thai_restaurant':           'Thai',
    'seafood_restaurant':        'Seafood',
    'fast_food_restaurant':      'Fast Food',
    'pizza_restaurant':          'Pizza',
    'hamburger_restaurant':      'Burgers',
    'vegetarian_restaurant':     'Vegetarian',
    'steak_house':               'Steakhouse',
    'sushi_restaurant':          'Sushi',
    'korean_restaurant':         'Korean',
    'vietnamese_restaurant':     'Vietnamese',
    'middle_eastern_restaurant': 'Middle Eastern',
    'mexican_restaurant':        'Mexican',
    'turkish_restaurant':        'Turkish',
    'lebanese_restaurant':       'Lebanese',
    'bakery':                    'Bakery',
    'cafe':                      'Café',
    'bar':                       'Bar & Grill',
    'meal_takeaway':             'Takeaway',
    'meal_delivery':             'Delivery',
    'ice_cream_shop':            'Desserts',
    'buffet_restaurant':         'Buffet',
    'breakfast_restaurant':      'Breakfast',
    'brunch_restaurant':         'Brunch',
    'coffee_shop':               'Coffee',
    'food_court':                'Food Court',
}

_PRICE_LABELS: Dict[int, str] = {
    0: 'Free',
    1: 'Budget',
    2: 'Moderate',
    3: 'Expensive',
    4: 'Luxury',
}

_NAMED_AREAS = [
    'Nugegoda', 'Dehiwala', 'Mount Lavinia', 'Borella', 'Pettah',
    'Maradana', 'Wellawatte', 'Bambalapitiya', 'Kollupitiya',
    'Cinnamon Gardens', 'Fort',
]


def extract_cuisine(types_str: Optional[str]) -> str:
    """Map a comma-separated Google Places types string to a cuisine label."""
    if not types_str:
        return 'Restaurant'
    tokens = [t.strip().lower() for t in types_str.split(',')]
    for token in tokens:
        if token in _CUISINE_MAP:
            return _CUISINE_MAP[token]
    return 'Restaurant'


def extract_location(address: Optional[str]) -> str:
    """Extract a Colombo district label from a Google Places address string."""
    if not address:
        return 'Colombo'
    match = re.search(r'Colombo\s+(\d+)', address, re.IGNORECASE)
    if match:
        return f'Colombo {int(match.group(1)):02d}'
    for area in _NAMED_AREAS:
        if area.lower() in address.lower():
            return area
    return 'Colombo'


def price_label(price_level) -> str:
    """Convert a numeric price_level (0–4) to a human-readable label."""
    try:
        return _PRICE_LABELS.get(int(price_level), 'Unknown')
    except (TypeError, ValueError):
        return 'Unknown'
