"""Synthetic house-price dataset generator (12,000 rows by default)."""
import numpy as np
import pandas as pd
from features import LOCATIONS

# True coefficients used to create the target (the model must rediscover them)
TRUE = dict(base=20000, area=110, bed=8000, bath=14000, floor=9000, age=-1200,
            garage=12000, d_center=-4500, d_school=-2500, crime=-900,
            cond=7000, pool=18000, garden=7000, furnish=6000)

def generate(n=12000, seed=42) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    loc = r.choice(list(LOCATIONS), n, p=[.12, .24, .08, .12, .10, .22, .12])
    bed = r.integers(1, 7, n)
    area = np.clip(bed * 350 + r.normal(300, 250, n), 350, 6000).round()
    bath = np.clip(np.round(bed * .7 + r.normal(0, .6, n)), 1, 6).astype(int)
    floors = r.choice([1, 2, 3], n, p=[.5, .4, .1])
    age = r.integers(0, 60, n)
    garage = r.choice([0, 1, 2, 3], n, p=[.15, .4, .35, .1])
    # city-centre distance depends on location
    centre_mean = {"Downtown": 2, "Uptown Elite": 5, "University": 4, "Riverside": 6,
                   "Suburb": 12, "Industrial": 10, "Rural": 28}
    d_center = np.clip(np.array([r.normal(centre_mean[l], 2) for l in loc]), .3, 50).round(1)
    d_school = np.clip(r.gamma(2, 1.2, n), .2, 15).round(1)
    crime = np.clip(r.normal(40, 18, n) + (loc == "Industrial") * 15
                    - (loc == "Uptown Elite") * 20, 1, 100).round(1)
    cond = r.integers(1, 11, n)
    pool = (r.random(n) < .12 + .3 * (loc == "Uptown Elite")).astype(int)
    garden = (r.random(n) < .45).astype(int)
    furn = r.choice([0, 1, 2], n, p=[.4, .4, .2])
    t = TRUE
    price = (t["base"] + t["area"] * area + t["bed"] * bed + t["bath"] * bath
             + t["floor"] * floors + t["age"] * age + t["garage"] * garage
             + t["d_center"] * d_center + t["d_school"] * d_school
             + t["crime"] * crime + t["cond"] * cond + t["pool"] * pool
             + t["garden"] * garden + t["furnish"] * furn
             + np.array([LOCATIONS[l][0] for l in loc])
             + r.normal(0, 20000, n))                       # market noise
    price = np.clip(price, 25000, None).round(-2)
    return pd.DataFrame(dict(location=loc, area_sqft=area, bedrooms=bed,
        bathrooms=bath, floors=floors, age_years=age, garage_spaces=garage,
        dist_center_km=d_center, dist_school_km=d_school, crime_index=crime,
        condition_score=cond, has_pool=pool, has_garden=garden,
        furnishing=furn, price=price))

if __name__ == "__main__":
    df = generate(); df.to_csv("house_data.csv", index=False)
    print(df.shape); print(df.describe().T.round(1))
