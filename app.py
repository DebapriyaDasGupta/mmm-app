
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Marketing Mix Modelling",
    page_icon="📊",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "date",
    "Sales",
    "TV_Spend",
    "Radio_Spend",
    "Digital_Spend",
]
CHANNELS = ["TV_Spend", "Radio_Spend", "Digital_Spend"]

st.title("📊 Marketing Mix Modelling")
st.caption("Explore sales trends, marketing investment, and modelled channel relationships.")

# --------------------------------------------------
# CSV upload and validation
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload marketing data (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV containing the required schema to get started.")
    st.code(",".join(REQUIRED_COLUMNS), language="text")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

df.columns = df.columns.str.strip()

missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV contains no data.")
    st.stop()

df = df[REQUIRED_COLUMNS].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["Sales", *CHANNELS]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df.isna().any().any():
    st.error("The data contains missing or invalid dates/numeric values. Please clean the CSV.")
    st.stop()

if df["date"].duplicated().any():
    st.error("Duplicate dates found. Aggregate the data to one row per date.")
    st.stop()

if (df[["Sales", *CHANNELS]] < 0).any().any():
    st.error("Sales and marketing spend must be non-negative.")
    st.stop()

df = df.sort_values("date").reset_index(drop=True)

if len(df) < 10:
    st.error("At least 10 observations are required for this starter model.")
    st.stop()

# --------------------------------------------------
# Data overview
# --------------------------------------------------
st.subheader("1. Data Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Observations", f"{len(df):,}")
col2.metric("Total Sales", f"{df['Sales'].sum():,.2f}")
col3.metric("Total Marketing Spend", f"{df[CHANNELS].sum().sum():,.2f}")

with st.expander("View uploaded data"):
    st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# Sales and spend trends
# --------------------------------------------------
st.subheader("2. Trends Over Time")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df["date"], df["Sales"], label="Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4))
for channel in CHANNELS:
    ax.plot(df["date"], df[channel], label=channel)
ax.set_xlabel("Date")
ax.set_ylabel("Marketing Spend")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Correlation analysis
# --------------------------------------------------
st.subheader("3. Correlation Analysis")

corr = df[["Sales", *CHANNELS]].corr()

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# Chronological regression model
# --------------------------------------------------
st.subheader("4. Marketing Mix Regression")

st.caption(
    "The model uses earlier observations for training and the latest "
    "20% for testing. This is a baseline linear model, not a causal MMM."
)

split_index = int(len(df) * 0.8)
if split_index < 3 or len(df) - split_index < 2:
    st.error("Not enough observations for a chronological train/test split.")
    st.stop()

X = df[CHANNELS]
y = df["Sales"]

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

try:
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    model = sm.OLS(y_train, X_train_const).fit()
    predictions = model.predict(X_test_const)

except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(f"Model fitting failed: {exc}")
    st.stop()

rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

m1, m2, m3 = st.columns(3)
m1.metric("Test RMSE", f"{rmse:,.2f}")
m2.metric("Test MAE", f"{mae:,.2f}")
m3.metric("Test R²", f"{r2:.3f}")

results = pd.DataFrame({
    "Date": df["date"].iloc[split_index:].values,
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions.values,
})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results["Date"], results["Actual Sales"], label="Actual")
ax.plot(results["Date"], results["Predicted Sales"], label="Predicted")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("View test predictions"):
    st.dataframe(results, use_container_width=True)

with st.expander("View regression summary"):
    st.text(model.summary().as_text())

# --------------------------------------------------
# Channel coefficients and contribution proxy
# --------------------------------------------------
st.subheader("5. Channel Analysis")

coefficients = model.params.reindex(CHANNELS)

st.write("Estimated channel coefficients")
st.dataframe(
    coefficients.rename("Coefficient").to_frame(),
    use_container_width=True,
)

st.caption(
    "A coefficient estimates the change in predicted sales associated "
    "with one additional spend unit, holding other included channels "
    "constant. Interpretation depends on units and model assumptions."
)

# Descriptive proxy: coefficient multiplied by observed channel spend.
# This is not a validated causal contribution estimate.
contribution = df[CHANNELS].mul(coefficients, axis="columns")
contribution.columns = [c.replace("_Spend", "") for c in CHANNELS]

st.write("Modelled channel contribution proxy (coefficient × spend)")
st.bar_chart(contribution.sum().rename("Contribution proxy"))

st.warning(
    "These are in-sample association-based proxies, not causal channel "
    "contributions or ROI estimates. Negative coefficients are possible. "
    "The baseline model does not account for adstock, saturation, "
    "seasonality, pricing, promotions, or other confounders."
)

# --------------------------------------------------
# Download outputs
# --------------------------------------------------
st.subheader("6. Export Results")

csv_data = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download test predictions",
    data=csv_data,
    file_name="mmm_predictions.csv",
    mime="text/csv",
)

st.success("Analysis completed.")