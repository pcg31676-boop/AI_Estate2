import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import joblib


"""Train and serve the XGBoost property valuation model."""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "property_valuation_model.joblib")
MODEL_FEATURE_SCHEMA_VERSION = 2
EMBEDDING_COLUMNS = [f"image_embedding_{index:03d}" for index in range(512)]
VISUAL_COLUMNS = [
    "image_quality", "image_brightness", "image_sharpness",
    "image_finish_score", "image_present",
    *EMBEDDING_COLUMNS,
]


def _normalize_property_id(property_id):
    identifier = str(property_id).strip()
    try:
        numeric_identifier = float(identifier)
    except ValueError:
        return identifier

    if numeric_identifier.is_integer():
        return str(int(numeric_identifier))
    return identifier


def make_sample_data(rows=320, seed=42):
    """Build reproducible labeled demo records when no local dataset is available."""
    rng = np.random.default_rng(seed)
    sqft = rng.integers(700, 4200, rows)
    bedrooms = rng.integers(1, 7, rows)
    bathrooms = np.round(rng.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], rows), 1)
    age = rng.integers(0, 100, rows)
    location_score = np.round(rng.uniform(1, 10, rows), 2)
    lat = 47.3 + rng.uniform(0, 0.35, rows)
    lon = -122.45 + rng.uniform(0, 0.35, rows)
    price = (
        55000 + sqft * 185 + bedrooms * 11500 + bathrooms * 9000
        + location_score * 18000 - age * 850 + rng.normal(0, 35000, rows)
    ).clip(90000)
    return pd.DataFrame({
        "id": [f"demo-{index + 1:04d}" for index in range(rows)],
        "price": price.round(0),
        "sqft_living": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_age": age,
        "location_score": location_score,
        "lat": lat,
        "long": lon,
    })


def _location_scores(data):
    if "location_score" in data:
        return pd.to_numeric(data["location_score"], errors="coerce").fillna(5.0)
    if "zip_wealth_rank" in data:
        neighborhood_values = pd.to_numeric(data["zip_wealth_rank"], errors="coerce")
        return neighborhood_values.rank(pct=True).fillna(0.5) * 10
    return pd.Series(5.0, index=data.index)


def _training_features(data, images_dir=None):
    numeric = data.drop(columns=[column for column in ("id", "price") if column in data])
    features = numeric.apply(pd.to_numeric, errors="coerce")
    features["location_score"] = _location_scores(data)
    visual_features = pd.DataFrame(0.0, index=features.index, columns=VISUAL_COLUMNS)
    features = pd.concat([features, visual_features], axis=1)

    if images_dir and "id" in data:
        image_rows = []
        for row_index, property_id in data["id"].items():
            folder = os.path.join(images_dir, _normalize_property_id(property_id))
            if not os.path.isdir(folder):
                continue
            paths = [
                os.path.join(folder, name)
                for name in sorted(os.listdir(folder))
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if paths:
                image_rows.append((row_index, paths))

        if image_rows:
            try:
                from vision_module import analyze_images

                for row_index, paths in image_rows:
                    analysis = analyze_images(paths)
                    for column in VISUAL_COLUMNS:
                        features.at[row_index, column] = analysis["features"].get(column, 0.0)
            except Exception as error:
                warnings.warn(
                    f"Could not extract training image features ({error}); "
                    "training will continue with tabular features.",
                    RuntimeWarning,
                )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return features


def train_model(data, images_dir=None, model_path=MODEL_PATH):
    if "price" not in data:
        raise ValueError("Training data must include a 'price' column.")

    prices = (
        data["price"].astype(str).str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False).str.strip()
    )
    prices = pd.to_numeric(prices, errors="coerce")
    valid_rows = prices.notna() & np.isfinite(prices)
    clean_data = data.loc[valid_rows].reset_index(drop=True)
    prices = prices.loc[valid_rows].reset_index(drop=True)
    if len(clean_data) < 10:
        raise ValueError("At least 10 valid labeled properties are required to train.")

    features = _training_features(clean_data, images_dir)
    x_train, x_test, y_train, y_test = train_test_split(
        features, prices, test_size=0.2, random_state=42
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        n_jobs=4,
        random_state=42,
    )
    model.fit(x_train, y_train, verbose=False)
    predictions = model.predict(x_test)
    absolute_errors = np.abs(y_test.to_numpy() - predictions)
    interval_error = float(np.quantile(absolute_errors, 0.9, method="higher"))
    bundle = {
        "model": model,
        "feature_columns": list(features.columns),
        "interval_error": interval_error,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "training_rows": len(clean_data),
        "feature_schema_version": MODEL_FEATURE_SCHEMA_VERSION,
    }
    joblib.dump(bundle, model_path)
    return bundle


def load_or_train_model(data, images_dir=None, model_path=MODEL_PATH):
    if os.path.isfile(model_path):
        try:
            bundle = joblib.load(model_path)
            if isinstance(bundle, dict) and {"model", "feature_columns", "interval_error"} <= bundle.keys():
                if (
                    bundle.get("feature_schema_version") == MODEL_FEATURE_SCHEMA_VERSION
                    and "image_finish_score" in bundle["feature_columns"]
                ):
                    return bundle
        except Exception:
            pass
    return train_model(data, images_dir=images_dir, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "properties.csv"))
    parser.add_argument("--images", default=os.path.join(BASE_DIR, "images"))
    parser.add_argument("--output", default=MODEL_PATH)
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="generate sample_properties.csv and train using the generated records",
    )
    args = parser.parse_args()

    if args.generate_sample:
        args.data = os.path.join(BASE_DIR, "sample_properties.csv")
        make_sample_data().to_csv(args.data, index=False)
        print(f"Generated demo dataset: {args.data}")

    data = pd.read_csv(args.data)
    bundle = train_model(data, images_dir=args.images, model_path=args.output)
    print(f"Model saved to: {args.output}")
    print(f"Training properties: {bundle['training_rows']}")
    print(f"Holdout MAE: {bundle['mae']:,.0f}")
    print(f"90% residual interval half-width: {bundle['interval_error']:,.0f}")


if __name__ == "__main__":
    main()