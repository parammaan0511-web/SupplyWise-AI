import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "day_of_week",
    "day_of_year",
    "lag_1",
    "lag_7",
    "mean_7",
    "mean_14",
]


def prepare_data(sales):
    df = sales.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"])

    # Modern pandas uses day_of_week and day_of_year
    if hasattr(df["date"].dt, "day_of_week"):
        df["day_of_week"] = df["date"].dt.day_of_week
    else:
        df["day_of_week"] = df["date"].dt.dayofweek

    if hasattr(df["date"].dt, "day_of_year"):
        df["day_of_year"] = df["date"].dt.day_of_year
    else:
        df["day_of_year"] = df["date"].dt.dayofyear

    grouped = df.groupby("product_id")["units_sold"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)

    df["mean_7"] = df.groupby("product_id")[
        "units_sold"
    ].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["mean_14"] = df.groupby("product_id")[
        "units_sold"
    ].transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    df = df.dropna(
        subset=FEATURES + ["units_sold"]
    ).copy()

    return df


def train_evaluate(sales):
    df = prepare_data(sales)

    split_date = df["date"].quantile(0.80)

    train = df[df["date"] <= split_date]
    test = df[df["date"] > split_date]

    if train.empty or test.empty:
        raise ValueError(
            "Not enough historical sales data."
        )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURES],
        train["units_sold"],
    )

    predictions = np.maximum(
        0,
        model.predict(test[FEATURES]),
    )

    baseline = np.maximum(
        0,
        test["mean_7"].to_numpy(),
    )

    metrics = {
        "Random Forest MAE": float(mean_absolute_error(
            test["units_sold"],
            predictions,
        )),
        "Random Forest RMSE": float(np.sqrt(
            mean_squared_error(
                test["units_sold"],
                predictions,
            )
        )),
        "Baseline MAE": float(mean_absolute_error(
            test["units_sold"],
            baseline,
        )),
        "Baseline RMSE": float(np.sqrt(
            mean_squared_error(
                test["units_sold"],
                baseline,
            )
        )),
    }

    evaluation = test[
        ["date", "product_id", "product_name", "units_sold"]
    ].copy()

    evaluation["predicted_units"] = predictions
    evaluation["baseline_units"] = baseline

    return model, metrics, evaluation, df


def forecast_product(model, prepared, product_id, horizon):
    history = prepared[
        prepared["product_id"] == product_id
    ].sort_values("date")

    if history.empty:
        raise ValueError(
            "No usable history for this product."
        )

    last_date = history["date"].max()

    recent = history["units_sold"].astype(float).tolist()

    forecasts = []

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)

        day_of_week = next_date.day_of_week if hasattr(next_date, "day_of_week") else next_date.dayofweek
        day_of_year = next_date.day_of_year if hasattr(next_date, "day_of_year") else next_date.dayofyear

        features = pd.DataFrame([{
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "lag_1": recent[-1],
            "lag_7": (
                recent[-7]
                if len(recent) >= 7
                else np.mean(recent)
            ),
            "mean_7": np.mean(recent[-7:]),
            "mean_14": np.mean(recent[-14:]),
        }])

        prediction = max(
            0,
            float(model.predict(features[FEATURES])[0]),
        )

        forecasts.append({
            "date": next_date,
            "forecast_units": round(prediction, 2),
        })

        recent.append(prediction)

    return pd.DataFrame(forecasts)
