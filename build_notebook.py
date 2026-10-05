"""Builds notebooks/capstone_sales_analysis.ipynb (run, then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Week 4 Capstone: Retail Sales Analysis and Prediction

**Objectives**
1. Perform Exploratory Data Analysis (EDA) on a retail sales dataset
2. Build a regression model to predict sales
3. Prepare data for a Power BI dashboard
4. Summarise findings and recommendations

> **Data note:** the capstone brief did not include a dataset, so this notebook uses a reproducible
> *synthetic* retail dataset (`src/generate_data.py`). To run on a real dataset (e.g. Kaggle Superstore),
> change `DATA_PATH` below and keep the same column names.
""")

md("## 1. Setup")
code("""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})

ROOT = Path("..").resolve()
DATA_PATH = ROOT / "data" / "retail_sales_raw.csv"
IMG = ROOT / "images"; IMG.mkdir(exist_ok=True)
PBI = ROOT / "powerbi"; PBI.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
RANDOM_STATE = 42
metrics_out = {}

def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / f"{name}.png", bbox_inches="tight"); plt.show()
""")

md("## 2. Load and Inspect")
code("""
raw = pd.read_csv(DATA_PATH)
print("Shape:", raw.shape)
raw.head()
""")
code("""
raw.info()
""")

md("## 3. Data Cleaning")
code("""
df = raw.copy()
metrics_out["rows_raw"] = int(len(df))

# 3.1 Duplicates
dup_count = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {dup_count}")

# 3.2 Missing values
missing = df.isna().sum()
print("Missing values before cleaning:\\n", missing[missing > 0])
for col in ["Segment", "Ship Mode"]:
    df[col] = df[col].fillna(df[col].mode()[0])   # categorical -> mode imputation
print("Missing values after cleaning:", int(df.isna().sum().sum()))

# 3.3 Data types and engineered features
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter"] = df["Order Date"].dt.quarter
df["DayOfWeek"] = df["Order Date"].dt.dayofweek
df["YearMonth"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
df["ShipDays"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["ProfitMargin"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"], 0)

metrics_out.update(rows_clean=int(len(df)), duplicates_removed=dup_count)
print("Clean shape:", df.shape)
""")

md("## 4. Exploratory Data Analysis")
md("### 4.1 Descriptive statistics")
code("""
df[["Quantity", "Discount", "Sales", "Profit", "ProfitMargin", "ShipDays"]].describe().round(2)
""")
code("""
total_sales, total_profit = df["Sales"].sum(), df["Profit"].sum()
kpis = {
    "total_sales": round(total_sales, 2),
    "total_profit": round(total_profit, 2),
    "profit_margin_pct": round(total_profit / total_sales * 100, 2),
    "orders": int(df["Order ID"].nunique()),
    "customers": int(df["Customer ID"].nunique()),
    "avg_order_value": round(df["Sales"].mean(), 2),
    "avg_discount_pct": round(df["Discount"].mean() * 100, 2),
    "loss_making_orders_pct": round((df["Profit"] < 0).mean() * 100, 2),
}
metrics_out["kpis"] = kpis
pd.Series(kpis)
""")

md("### 4.2 Sales distribution")
code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Sales"], bins=60, kde=True, ax=ax[0]); ax[0].set_title("Sales per order (raw)")
sns.histplot(np.log1p(df["Sales"].clip(lower=0)), bins=60, kde=True, ax=ax[1], color="C1")
ax[1].set_title("Sales per order (log scale)")
save(fig, "01_sales_distribution")
print("Skewness (raw):", round(df["Sales"].skew(), 2))
""")
md("Sales is strongly right-skewed (many small orders, a few very large ones), so the model is trained on a log-transformed target.")

md("### 4.3 Sales trend and seasonality")
code("""
monthly = df.groupby("YearMonth")[["Sales", "Profit"]].sum()
fig, ax = plt.subplots(figsize=(12, 4.5))
monthly["Sales"].plot(ax=ax, label="Sales", lw=2)
monthly["Profit"].plot(ax=ax, label="Profit", lw=2)
ax.set_title("Monthly Sales and Profit"); ax.set_ylabel("USD"); ax.legend()
save(fig, "02_monthly_trend")

yearly = df.groupby("Year")[["Sales", "Profit"]].sum()
yearly["YoY Sales Growth %"] = (yearly["Sales"].pct_change() * 100).round(1)
seasonal = df.groupby("Month")["Sales"].sum()
metrics_out["yearly"] = yearly.round(2).reset_index().to_dict("records")
metrics_out["peak_months"] = [int(m) for m in seasonal.nlargest(3).index]
yearly.round(2)
""")

md("### 4.4 Category and sub-category performance")
code("""
cat = df.groupby("Category")[["Sales", "Profit"]].sum()
cat["Margin %"] = (cat["Profit"] / cat["Sales"] * 100).round(1)
sub = df.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]].sum()
sub["Margin %"] = (sub["Profit"] / sub["Sales"] * 100).round(1)
sub = sub.sort_values("Profit")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
cat[["Sales", "Profit"]].plot.bar(ax=ax[0], rot=0); ax[0].set_title("Sales and Profit by Category")
colors = ["#c0392b" if p < 0 else "#2e86c1" for p in sub["Profit"]]
sub["Profit"].plot.barh(ax=ax[1], color=colors)
ax[1].set_title("Profit by Sub-Category"); ax[1].set_ylabel("")
save(fig, "03_category_performance")
metrics_out["category"] = cat.round(2).reset_index().to_dict("records")
metrics_out["subcategory"] = sub.round(2).reset_index().to_dict("records")
cat
""")
code("""
sub.round(2)
""")

md("### 4.5 Regional and segment analysis")
code("""
reg = df.groupby("Region")[["Sales", "Profit"]].sum()
reg["Margin %"] = (reg["Profit"] / reg["Sales"] * 100).round(1)
seg = df.groupby("Segment")[["Sales", "Profit"]].sum()
seg["Margin %"] = (seg["Profit"] / seg["Sales"] * 100).round(1)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
reg[["Sales", "Profit"]].plot.bar(ax=ax[0], rot=0); ax[0].set_title("Sales and Profit by Region")
seg[["Sales", "Profit"]].plot.bar(ax=ax[1], rot=0); ax[1].set_title("Sales and Profit by Segment")
save(fig, "04_region_segment")
metrics_out["region"] = reg.round(2).reset_index().to_dict("records")
metrics_out["segment"] = seg.round(2).reset_index().to_dict("records")
display(reg); display(seg)
""")

md("### 4.6 Impact of discounting")
code("""
disc = df.groupby("Discount").agg(Orders=("Sales", "size"), Sales=("Sales", "sum"),
                                  Profit=("Profit", "sum"), AvgMargin=("ProfitMargin", "mean"))
disc["AvgMargin %"] = (disc["AvgMargin"] * 100).round(1)
disc = disc.drop(columns="AvgMargin")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=df, x="Discount", y="ProfitMargin", ax=ax[0], showfliers=False)
ax[0].axhline(0, color="red", ls="--", lw=1); ax[0].set_title("Profit margin by discount level")
disc["Profit"].plot.bar(ax=ax[1], color=["#c0392b" if p < 0 else "#2e86c1" for p in disc["Profit"]], rot=0)
ax[1].set_title("Total profit by discount level")
save(fig, "05_discount_impact")

breakeven = df.groupby("Discount")["ProfitMargin"].mean()
metrics_out["discount"] = disc.round(2).reset_index().to_dict("records")
metrics_out["first_negative_discount"] = float(breakeven[breakeven < 0].index.min()) if (breakeven < 0).any() else None
metrics_out["first_loss_discount"] = float(disc.index[disc["Profit"] < 0].min())
metrics_out["profit_from_discount_ge_20"] = round(float(df.loc[df["Discount"] >= 0.20, "Profit"].sum()), 2)
metrics_out["orders_share_discount_ge_20_pct"] = round(float((df["Discount"] >= 0.20).mean() * 100), 1)
disc
""")

md("### 4.7 Correlation analysis")
code("""
num_cols = ["Quantity", "Discount", "Sales", "Profit", "ProfitMargin", "ShipDays"]
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix")
save(fig, "06_correlation_heatmap")
""")

md("### 4.8 Customers and shipping")
code("""
top_cust = (df.groupby("Customer ID")["Sales"].sum().sort_values(ascending=False))
top10_share = top_cust.head(int(len(top_cust) * 0.10)).sum() / top_cust.sum() * 100
metrics_out["top10pct_customer_sales_share"] = round(float(top10_share), 1)
print(f"Top 10% of customers generate {top10_share:.1f}% of sales")
ship = df.groupby("Ship Mode").agg(Orders=("Sales", "size"), Sales=("Sales", "sum"), AvgShipDays=("ShipDays", "mean")).round(2)
ship
""")

md("""
## 5. Regression Model: Predicting Sales

**Target:** `Sales` (modelled as `log1p(Sales)`, then converted back to USD for evaluation).
**Features:** Quantity, Discount, Category, Sub-Category, Region, Segment, Ship Mode, Year, Month, Day of week.

Models compared: Linear Regression (baseline), Random Forest, Gradient Boosting.
""")
code("""
features_num = ["Quantity", "Discount", "Year", "Month", "DayOfWeek"]
features_cat = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
X = df[features_num + features_cat]
y = np.log1p(df["Sales"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, " Test:", X_test.shape)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ("num", "passthrough", features_num),
])
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE),
}

results, fitted = [], {}
for name, est in models.items():
    pipe = Pipeline([("pre", pre), ("model", est)]).fit(X_train, y_train)
    pred_usd = np.expm1(pipe.predict(X_test)); true_usd = np.expm1(y_test)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
    results.append({
        "Model": name,
        "R2 (test, USD)": r2_score(true_usd, pred_usd),
        "R2 (test, log)": r2_score(y_test, pipe.predict(X_test)),
        "MAE (USD)": mean_absolute_error(true_usd, pred_usd),
        "RMSE (USD)": np.sqrt(mean_squared_error(true_usd, pred_usd)),
        "CV R2 (log, 5-fold)": cv,
    })
    fitted[name] = pipe

res = pd.DataFrame(results).set_index("Model").round(3)
metrics_out["model_comparison"] = res.reset_index().to_dict("records")
res
""")
code("""
best_name = res["R2 (test, USD)"].idxmax()
best = fitted[best_name]
metrics_out["best_model"] = best_name
print("Best model:", best_name)

pred = np.expm1(best.predict(X_test)); actual = np.expm1(y_test)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(actual, pred, alpha=0.35, s=12)
lim = [0, max(actual.max(), pred.max())]
ax[0].plot(lim, lim, "r--", lw=1); ax[0].set_xlabel("Actual Sales"); ax[0].set_ylabel("Predicted Sales")
ax[0].set_title(f"Actual vs Predicted ({best_name})")
sns.histplot(pred - actual, bins=60, ax=ax[1], color="C2"); ax[1].set_title("Residuals (Predicted - Actual, USD)")
ax[1].set_xlim(-1500, 1500)
save(fig, "07_model_evaluation")
""")
md("### 5.1 Feature importance")
code("""
names = best.named_steps["pre"].get_feature_names_out()
model = best.named_steps["model"]
imp = pd.Series(model.feature_importances_, index=[n.split("__")[1] for n in names])
# collapse one-hot columns back to their parent feature
parent = lambda n: next((c for c in features_cat if n.startswith(c + "_")), n)
imp_grouped = imp.groupby(parent).sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
imp_grouped.plot.barh(ax=ax, color="#2e86c1"); ax.set_title(f"Feature Importance ({best_name})")
save(fig, "08_feature_importance")
metrics_out["feature_importance"] = imp_grouped.sort_values(ascending=False).round(4).to_dict()
imp_grouped.sort_values(ascending=False).round(4)
""")
md("""
### 5.2 Interpretation
- Quantity and product sub-category drive order value; discount reduces revenue per unit but is not a strong predictor of order size.
- The model captures large-order behaviour less precisely (heteroscedastic errors), which is typical for sales data.
- Year and month capture growth and seasonality only modestly at the order level; use a time-series model if monthly forecasting is the goal.
""")

md("## 6. Export Data for Power BI")
code("""
fact = df[["Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Segment", "Region", "State",
           "Category", "Sub-Category", "Quantity", "Discount", "Sales", "Profit"]].copy()
fact["Profit Margin"] = np.where(fact["Sales"] > 0, fact["Profit"] / fact["Sales"], 0).round(4)
fact["Is Loss"] = (fact["Profit"] < 0).astype(int)
fact.to_csv(PBI / "sales_clean_for_powerbi.csv", index=False)

# Model predictions on the test set (for a "model accuracy" visual)
pred_df = df.loc[X_test.index, ["Order ID", "Order Date", "Category", "Sub-Category", "Region", "Sales"]].copy()
pred_df["Predicted Sales"] = pred.round(2)
pred_df["Abs Error"] = (pred_df["Predicted Sales"] - pred_df["Sales"]).abs().round(2)
pred_df.to_csv(PBI / "model_predictions_for_powerbi.csv", index=False)

df.to_csv(ROOT / "data" / "retail_sales_clean.csv", index=False)
with open(REPORTS / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2, default=str)
print("Exports written:", [p.name for p in PBI.glob("*.csv")])
""")

md("## 7. Key Findings and Recommendations")
code("""
from IPython.display import Markdown
k = metrics_out["kpis"]
worst_sub = sub.index[0][1]; best_sub = sub.index[-1][1]
top_region = reg["Sales"].idxmax(); low_margin_region = reg["Margin %"].idxmin()
Markdown(f'''
**Headline KPIs:** Sales **${k["total_sales"]:,.0f}**, Profit **${k["total_profit"]:,.0f}** ({k["profit_margin_pct"]}% margin), {k["orders"]:,} orders, {k["customers"]:,} customers.

1. **Growth and seasonality:** sales grow every year, with peaks in months {metrics_out["peak_months"]}.
2. **Profitability:** best sub-category by profit is **{best_sub}**; weakest is **{worst_sub}**; {k["loss_making_orders_pct"]}% of orders lose money.
3. **Discounting:** total profit turns negative from a {metrics_out["first_loss_discount"]:.0%} discount, and the average order margin turns negative at {metrics_out["first_negative_discount"]:.0%}. Orders discounted 20% or more ({metrics_out["orders_share_discount_ge_20_pct"]}% of orders) earned **${metrics_out["profit_from_discount_ge_20"]:,.0f}** in total profit.
4. **Geography:** **{top_region}** leads on sales; **{low_margin_region}** has the lowest margin.
5. **Model:** **{best_name}** achieves R2 = {res.loc[best_name, "R2 (test, USD)"]} and MAE = ${res.loc[best_name, "MAE (USD)"]:,.0f} on held-out data.
''')
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
Path(__file__).resolve().parents[1].joinpath("notebooks", "capstone_sales_analysis.ipynb").write_text(nbf.writes(nb))
print("Notebook built.")
