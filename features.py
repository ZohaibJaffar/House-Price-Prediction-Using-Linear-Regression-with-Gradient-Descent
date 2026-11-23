"""Shared feature definitions + preprocessing (used by trainer and UI)."""
import numpy as np
import pandas as pd

# location -> (additive price premium in $, description)
LOCATIONS = {
    "Rural":        (0,      "Village / countryside"),
    "Suburb":       (25000,  "Quiet residential suburb"),
    "Industrial":   (-5000,  "Near factories / warehouses"),
    "University":   (40000,  "Near campus, high rental demand"),
    "Riverside":    (70000,  "Scenic waterfront"),
    "Downtown":     (95000,  "City centre"),
    "Uptown Elite": (150000, "Premium gated neighbourhood"),
}
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "floors", "age_years",
           "garage_spaces", "dist_center_km", "dist_school_km",
           "crime_index", "condition_score", "has_pool", "has_garden",
           "furnishing"]
LOC_COLS = [f"loc_{n}" for n in list(LOCATIONS)[1:]]   # "Rural" = baseline
FEATURES = NUMERIC + LOC_COLS
TARGET = "price"

def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """DataFrame (with a 'location' column) -> numeric design matrix."""
    X = df[NUMERIC].copy()
    for name in list(LOCATIONS)[1:]:
        X[f"loc_{name}"] = (df["location"] == name).astype(float)
    return X[FEATURES].to_numpy(dtype=float)
