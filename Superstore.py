import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error

# Importing the Dataset
df = pd.read_csv("sample_-_superstore.csv")

# Creating a Quarter feature to control for seasonal effect
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str) + " " + df["Order Date"].dt.year.astype(str)
margin = df['Profit'] / df['Sales']

# Creating the features used in the models
cat = df[["Sub-Category", "Segment", "Region", "Ship Mode", "quarter"]]
cat_cols = list(cat.columns)
X = pd.concat([df['Discount'], cat], axis=1)
y = margin

# Transforming categorical data into variables
pre = ColumnTransformer([
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

# Creating the pipelines for OLS and Lasso (and Finetuning for lasso)
ols = Pipeline([("pre", pre), ("model", LinearRegression())])
lasso = Pipeline(steps=[("pre", pre), ("model", LassoCV(
    cv=15,
    random_state=42,
    max_iter=1000,
    n_alphas=100
))])

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Runnig both pipelines
for name, model in [("OLS", ols), ("Lasso", lasso)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "R2:", r2_score(y_test, pred), "MAE:", mean_absolute_error(y_test, pred))

elas_pre = ColumnTransformer(
    transformers=[
        ("log_discount", FunctionTransformer(np.log1p, validate=False), ["Discount"]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

ols_elasticity = Pipeline([("pre", elas_pre), ("model", LinearRegression())])

y_elasticity = np.log(df["Sales"].values)
X_elasticity = pd.concat([df[["Discount"]], df[cat_cols]], axis=1)
ols_elasticity.fit(X_elasticity, y_elasticity)

feat_names = ( ["log_price_change__Discount"]+ list(elas_pre.named_transformers_["cat"].get_feature_names_out(cat_cols)))
coef = ols_elasticity.named_steps["model"].coef_
elasticity = float(coef[list(feat_names).index("log_price_change__Discount")])
print("Estimated price elasticity of sales:", elasticity)


def simulate_discount_change(df_in, new_discount_cap):
    """
    Simulate the impact of capping discounts at a certain level.

    Returns: predicted_profit (total profit under new discount policy)
    """
    df_sim = df_in.copy()

    # Cap the discount
    df_sim["Discount_new"] = df_sim["Discount"].clip(upper=new_discount_cap)

    # Calculate discount change (negative when reducing discount)
    discount_change = df_sim["Discount_new"] - df_sim["Discount"]

    # Predict new margin based on capped discount
    X_sim = X.copy()
    X_sim["Discount"] = df_sim["Discount_new"]

    predicted_margin = lasso.predict(X_sim)

    # Calculate sales impact from discount change
    sales_multiplier = -elasticity * discount_change
    predicted_sales = np.where(df["Discount"] > new_discount_cap,df["Sales"] * sales_multiplier,df["Sales"]                 )

    # Calculate new profit
    predicted_profit = predicted_margin * predicted_sales

    return predicted_profit, predicted_sales, predicted_margin



# Simulate different discount cap scenarios
print("\n" + "=" * 60)
print("DISCOUNT SIMULATION RESULTS")
print("=" * 60)

# Calculate baseline (current state)
baseline_profit = df["Profit"].sum()
baseline_sales = df["Sales"].sum()
baseline_margin = baseline_profit / baseline_sales

print(f"\nBASELINE (Current State):")
print(f"  Total Profit: ${baseline_profit:,.2f}")
print(f"  Total Sales: ${baseline_sales:,.2f}")
print(f"  Avg Margin: {baseline_margin:.2%}")

# Test different discount caps
discount_scenarios = np.round(np.arange(0.00, 0.80 + 0.01, 0.01), 2)
caps_out = []
profit_out = []
sales_out = []


print(f"\nSIMULATION SCENARIOS:")
print(
    f"{'Discount Cap':<15} {'Total Profit':<20} {'Change $':<20} {'Change %':<15} {'Total Sales':<20} {'Sales Change %':<15}")
print("-" * 120)

for cap in discount_scenarios:
    pred_profit, pred_sales, pred_margin = simulate_discount_change(df, cap)

    total_pred_profit = pred_profit.sum()
    total_pred_sales = pred_sales.sum()

    profit_change = total_pred_profit - baseline_profit
    profit_change_pct = (profit_change / baseline_profit) * 100
    sales_change_pct = ((total_pred_sales - baseline_sales) / baseline_sales) * 100

    caps_out.append(cap)
    profit_out.append(total_pred_profit)
    sales_out.append(total_pred_sales)

    print(
        f"{cap:<15.0%} ${total_pred_profit:<19,.2f} ${profit_change:<19,.2f} {profit_change_pct:<14.2f}% ${total_pred_sales:<19,.2f} {sales_change_pct:<14.2f}%")
results_df = pd.DataFrame({
    "discount_cap": caps_out,
    "total_profit": profit_out,
    "total_sales": sales_out
})
results_df["profit_change_$"] = results_df["total_profit"] - baseline_profit
results_df["profit_change_%"] = 100 * (results_df["total_profit"] - baseline_profit) / baseline_profit
results_df["sales_change_%"]  = 100 * (results_df["total_sales"] - baseline_sales) / baseline_sales

# Round for readability
results_df = results_df.round(2)

# Save to CSV
csv_path = "discount_cap_simulation_results.csv"
results_df.to_csv(csv_path, index=False)

print(f"\n✅ Results saved to: {csv_path}")

x = np.asarray(caps_out).ravel()
y_profit = np.asarray(profit_out).ravel()
y_sales  = np.asarray(sales_out).ravel()

idx_max = np.argmax(y_profit)
x_max = x[idx_max]
y_max = y_profit[idx_max]


print("\n" + "=" * 60)
print(f"Note: Current average discount in dataset: {df['Discount'].mean():.2%}")
print(f"Transactions with discount > 0: {(df['Discount'] > 0).sum()} out of {len(df)}")

plt.plot(x, y_profit, label="Profit ($)")
plt.plot(x, y_sales,  label="Sales ($)", linestyle="--")
plt.scatter(x_max, y_max, color="red", s=80, zorder=5, label=f"Max Profit = ${y_max:,.0f} at {np.round(x_max * 100)}% cap")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
def currency_fmt(x, pos):
    return f'${x:,.0f}'  # e.g., 1234567 → $1,234,568
plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_fmt))
plt.xlabel("Discount Cap (%)", )
plt.ylabel("Value ($)")
plt.title("Profit & Sales under Discount Caps")
plt.legend()
plt.grid(True)
plt.show()
