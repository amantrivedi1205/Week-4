"""
Generate a reproducible synthetic retail sales dataset (Superstore-style).

The capstone brief did not ship with a dataset, so this script creates one with
realistic structure: seasonality, category price levels, discount effects on
profit, regional mix, and deliberate data-quality issues (missing values,
duplicates) so the EDA / cleaning steps have something real to do.

To use a real dataset instead (e.g. Kaggle "Superstore"), skip this script and
point the notebook's DATA_PATH at your CSV, keeping the same column names.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 9000
OUT = Path(__file__).resolve().parents[1] / "data" / "retail_sales_raw.csv"

rng = np.random.default_rng(SEED)

# (category, sub_category, base_price, base_margin)
CATALOGUE = [
    ("Technology", "Phones", 420, 0.18), ("Technology", "Accessories", 45, 0.30),
    ("Technology", "Copiers", 900, 0.28), ("Technology", "Machines", 650, 0.05),
    ("Furniture", "Chairs", 280, 0.10), ("Furniture", "Tables", 520, -0.04),
    ("Furniture", "Bookcases", 310, 0.02), ("Furniture", "Furnishings", 60, 0.22),
    ("Office Supplies", "Binders", 25, 0.30), ("Office Supplies", "Paper", 18, 0.38),
    ("Office Supplies", "Storage", 85, 0.12), ("Office Supplies", "Appliances", 130, 0.20),
]
CAT_P = np.array([.07, .12, .03, .02, .06, .04, .04, .08, .16, .18, .12, .08])
REGIONS = {"West": 0.32, "East": 0.28, "Central": 0.22, "South": 0.18}
REGION_FACTOR = {"West": 1.08, "East": 1.04, "Central": 0.96, "South": 0.92}
SEGMENTS = {"Consumer": 0.52, "Corporate": 0.30, "Home Office": 0.18}
SEG_FACTOR = {"Consumer": 0.97, "Corporate": 1.06, "Home Office": 1.00}
SHIP = {"Standard Class": 0.58, "Second Class": 0.20, "First Class": 0.15, "Same Day": 0.07}
SHIP_DAYS = {"Standard Class": 5, "Second Class": 3, "First Class": 2, "Same Day": 0}
STATES = {
    "West": ["California", "Washington", "Oregon", "Arizona"],
    "East": ["New York", "Pennsylvania", "Massachusetts", "Virginia"],
    "Central": ["Texas", "Illinois", "Ohio", "Michigan"],
    "South": ["Florida", "Georgia", "North Carolina", "Tennessee"],
}

# Order dates with seasonality + yearly growth
days = pd.date_range("2022-01-01", "2025-12-31", freq="D")
month_w = {1: .75, 2: .75, 3: .9, 4: .9, 5: .95, 6: 1.0, 7: .95, 8: 1.0,
           9: 1.2, 10: 1.1, 11: 1.45, 12: 1.5}
year_w = {2022: 0.85, 2023: 1.0, 2024: 1.12, 2025: 1.25}
w = np.array([month_w[d.month] * year_w[d.year] for d in days])
order_dates = pd.to_datetime(np.sort(rng.choice(days, size=N_ORDERS, p=w / w.sum())))

cat_idx = rng.choice(len(CATALOGUE), size=N_ORDERS, p=CAT_P)
regions = rng.choice(list(REGIONS), size=N_ORDERS, p=list(REGIONS.values()))
segments = rng.choice(list(SEGMENTS), size=N_ORDERS, p=list(SEGMENTS.values()))
ship_mode = rng.choice(list(SHIP), size=N_ORDERS, p=list(SHIP.values()))
discount = rng.choice([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50], size=N_ORDERS,
                      p=[.42, .10, .14, .08, .14, .06, .04, .02])
quantity = np.clip(rng.poisson(2.6, N_ORDERS) + 1, 1, 14)

rows = []
for i in range(N_ORDERS):
    cat, sub, base_price, base_margin = CATALOGUE[cat_idx[i]]
    region, seg = regions[i], segments[i]
    unit_price = base_price * REGION_FACTOR[region] * SEG_FACTOR[seg] * rng.lognormal(0, 0.12)
    sales = unit_price * quantity[i] * (1 - discount[i])
    margin = base_margin - 0.9 * discount[i] + rng.normal(0, 0.04)  # discounts erode margin
    ship_date = order_dates[i] + pd.Timedelta(days=int(SHIP_DAYS[ship_mode[i]] + rng.integers(0, 2)))
    rows.append({
        "Order ID": f"ORD-{order_dates[i].year}-{i + 1:05d}",
        "Order Date": order_dates[i].date(), "Ship Date": ship_date.date(),
        "Ship Mode": ship_mode[i], "Customer ID": f"CUST-{rng.integers(1, 1800):04d}",
        "Segment": seg, "Region": region, "State": rng.choice(STATES[region]),
        "Category": cat, "Sub-Category": sub, "Quantity": int(quantity[i]),
        "Discount": float(discount[i]), "Sales": round(float(sales), 2),
        "Profit": round(float(sales * margin), 2),
    })

df = pd.DataFrame(rows)

# Inject realistic data-quality issues
df.loc[rng.choice(df.index, size=int(0.012 * len(df)), replace=False), "Segment"] = np.nan
df.loc[rng.choice(df.index, size=int(0.008 * len(df)), replace=False), "Ship Mode"] = np.nan
dupes = df.sample(n=45, random_state=SEED)
df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df):,} rows -> {OUT}")
