"""
Predictive Analytics Using Historical Data
Regression + Time-Series forecasting, with cleaning, evaluation and visualization.

Usage:
    python predictive_analytics.py                 # uses built-in sample data
    python predictive_analytics.py mydata.csv date value   # your own CSV (date column, value column)
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- 1. LOAD DATA
def load_data():
    if len(sys.argv) >= 4:
        df = pd.read_csv(sys.argv[1])
        df = df.rename(columns={sys.argv[2]: "date", sys.argv[3]: "value"})[["date", "value"]]
    else:
        # Sample: 6 years of monthly sales = trend + yearly seasonality + noise
        rng = np.random.default_rng(42)
        dates = pd.date_range("2018-01-01", periods=72, freq="MS")
        t = np.arange(72)
        value = 200 + 3.5 * t + 30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 8, 72)
        df = pd.DataFrame({"date": dates, "value": value})
        # add dirty data to demonstrate cleaning
        df.loc[[10, 25, 40], "value"] = np.nan
        df.loc[55, "value"] = 900
        df = pd.concat([df, df.iloc[[5]]])  # duplicate row
    return df


# ------------------------------------------------------------- 2. CLEAN DATA
def clean(df):
    print("Raw shape:", df.shape, "| Missing:", int(df["value"].isna().sum()))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset="date")
    df = df.sort_values("date").set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # outliers: IQR method -> mark as NaN
    q1, q3 = df["value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (df["value"] < q1 - 3 * iqr) | (df["value"] > q3 + 3 * iqr)
    print("Outliers removed:", int(mask.sum()))
    df.loc[mask, "value"] = np.nan

    # fill gaps by time interpolation
    df = df.asfreq(pd.infer_freq(df.index) or "MS")
    df["value"] = df["value"].interpolate(method="time").bfill().ffill()
    print("Clean shape:", df.shape)
    return df


# ------------------------------------------------------ 3. FEATURE ENGINEERING
def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["t"] = np.arange(len(df))
    f["month"] = df.index.month
    month_dummies = pd.get_dummies(f["month"], prefix="m", drop_first=True).astype(float)
    return pd.concat([f[["t"]], month_dummies], axis=1)


def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:<22} MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:6.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE%": mape}


def main():
    df = clean(load_data())

    # ------------------------------------------------- 4. TRAIN / TEST SPLIT
    test_size = max(6, int(len(df) * 0.2))     # time-based split (no shuffling!)
    train, test = df.iloc[:-test_size], df.iloc[-test_size:]

    # --------------------------------------------- 5A. REGRESSION (trend+season)
    X = make_features(df)
    Xtr, Xte = X.iloc[:-test_size], X.iloc[-test_size:]
    lr = LinearRegression().fit(Xtr, train["value"])
    pred_lr = lr.predict(Xte)

    # --------------------------------------------- 5B. TIME SERIES (Holt-Winters)
    seasonal = 12 if len(train) >= 24 else None
    hw = ExponentialSmoothing(train["value"], trend="add",
                              seasonal="add" if seasonal else None,
                              seasonal_periods=seasonal).fit()
    pred_hw = hw.forecast(test_size).values

    # naive baseline = last observed value
    pred_naive = np.repeat(train["value"].iloc[-1], test_size)

    # ------------------------------------------------------- 6. EVALUATION
    print("\n=== Model Evaluation (test set) ===")
    results = pd.DataFrame([
        metrics(test["value"].values, pred_naive, "Naive baseline"),
        metrics(test["value"].values, pred_lr, "Linear Regression"),
        metrics(test["value"].values, pred_hw, "Holt-Winters"),
    ]).set_index("model")
    best = results["RMSE"].idxmin()
    print("Best model:", best)

    # ----------------------------------------------- 7. FUTURE FORECAST (12)
    horizon = 12
    future_idx = pd.date_range(df.index[-1], periods=horizon + 1, freq=df.index.freq)[1:]
    if best == "Linear Regression":
        lr_full = LinearRegression().fit(X, df["value"])
        full = pd.DataFrame(index=df.index.append(future_idx))
        full["value"] = np.nan
        Xf = make_features(full).reindex(columns=X.columns, fill_value=0.0)
        future = lr_full.predict(Xf.iloc[-horizon:])
    else:
        hw_full = ExponentialSmoothing(df["value"], trend="add",
                                       seasonal="add" if seasonal else None,
                                       seasonal_periods=seasonal).fit()
        future = hw_full.forecast(horizon).values
    future = pd.Series(future, index=future_idx, name="forecast")
    future.to_csv("future_forecast.csv")

    # ------------------------------------------------------ 8. VISUALIZATION
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # (1) Historical trend
    sns.lineplot(x=df.index, y=df["value"], ax=ax[0, 0], color="black", marker="o")
    ax[0, 0].set_title("1. Historical Data (Trend)")
    ax[0, 0].set_xlabel("Date"); ax[0, 0].set_ylabel("Sales")

    # (2) Test-period predictions vs actual
    sns.lineplot(x=train.index, y=train["value"], ax=ax[0, 1], color="grey", label="Train")
    sns.lineplot(x=test.index, y=test["value"], ax=ax[0, 1], color="black", marker="o", label="Actual")
    sns.lineplot(x=test.index, y=pred_lr, ax=ax[0, 1], marker="s", label="Linear Regression")
    sns.lineplot(x=test.index, y=pred_hw, ax=ax[0, 1], marker="^", label="Holt-Winters")
    ax[0, 1].set_title("2. Actual vs Predicted (Test Set)")
    ax[0, 1].set_xlabel("Date"); ax[0, 1].set_ylabel("Sales")

    # (3) Model comparison (RMSE)
    sns.barplot(x=results.index, y=results["RMSE"], ax=ax[1, 0], palette="viridis")
    ax[1, 0].set_title("3. Model Comparison (lower RMSE = better)")
    ax[1, 0].set_xlabel(""); ax[1, 0].set_ylabel("RMSE")
    for p in ax[1, 0].patches:
        ax[1, 0].annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                          ha="center", va="bottom")

    # (4) Future forecast
    sns.lineplot(x=df.index, y=df["value"], ax=ax[1, 1], color="black", label="Historical")
    sns.lineplot(x=future.index, y=future.values, ax=ax[1, 1], color="red", marker="o",
                 label=f"Forecast ({best})")
    ax[1, 1].set_title(f"4. Next {horizon} Months Forecast")
    ax[1, 1].set_xlabel("Date"); ax[1, 1].set_ylabel("Sales")

    for a in ax.flat:
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("forecast_plot.png", dpi=150)   # copy for your report
    print("\nSaved: forecast_plot.png, future_forecast.csv")
    print(future.round(2))
    plt.show()                                  # graph window popup


if __name__ == "__main__":
    main()
