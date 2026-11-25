"""Train a dedicated land price-per-square-foot model from labeled land data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
LAND_MODEL_PATH = BASE_DIR / "land_valuation_model.pkl"
LAND_MODEL_SCHEMA_VERSION = 2
LAND_NUMERIC_FEATURES = [
    "plot_area_sqft",
    "frontage_width_ft",
    "corner_plot_flag",
    "fsi_far_ratio",
    "road_width_ft",
    "distance_to_main_road_m",
    "water_power_utility_access",
    "lat",
    "long",
    "micro_location_score",
]
LAND_CATEGORICAL_FEATURES = [
    "topography_type",
    "soil_type",
    "zoning_category",
    "access_road_type",
]
LAND_FEATURES = LAND_NUMERIC_FEATURES + LAND_CATEGORICAL_FEATURES
LAND_TARGET = "land_price_per_sqft"


def make_synthetic_land_data(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate clearly synthetic Bengaluru-like land records for pipeline demos."""
    if rows < 30:
        raise ValueError("Synthetic data generation requires at least 30 rows.")
    rng = np.random.default_rng(seed)
    zoning = rng.choice(
        ["Residential", "Commercial", "Industrial", "Agricultural", "Mixed-Use"],
        rows,
        p=[0.42, 0.18, 0.12, 0.16, 0.12],
    )
    topography = rng.choice(
        ["Flat", "Sloped", "Irregular"], rows, p=[0.68, 0.18, 0.14]
    )
    soil = rng.choice(["Red", "Clay", "Sandy", "Black"], rows)
    access_road = rng.choice(
        ["Paved / National Highway", "Tar Road", "Dirt Road"],
        rows,
        p=[0.25, 0.60, 0.15],
    )
    plot_area = rng.integers(600, 15001, rows)
    frontage = rng.uniform(15, 150, rows)
    corner = rng.integers(0, 2, rows)
    fsi_far = rng.uniform(0.5, 4.0, rows)
    road_width = rng.uniform(10, 100, rows)
    distance_to_road = rng.uniform(0, 2500, rows)
    utilities = rng.integers(0, 2, rows)
    latitude = rng.uniform(12.75, 13.15, rows)
    longitude = rng.uniform(77.45, 77.85, rows)
    micro_score = rng.uniform(1, 10, rows)

    zoning_effect = pd.Series(zoning).map(
        {
            "Residential": 0,
            "Commercial": 9000,
            "Industrial": 1800,
            "Agricultural": -2200,
            "Mixed-Use": 7000,
        }
    ).to_numpy()
    topography_effect = pd.Series(topography).map(
        {"Flat": 0, "Sloped": -2500, "Irregular": -4000}
    ).to_numpy()
    soil_effect = pd.Series(soil).map(
        {"Red": 900, "Clay": -500, "Sandy": -900, "Black": 300}
    ).to_numpy()
    access_effect = pd.Series(access_road).map(
        {
            "Paved / National Highway": 5000,
            "Tar Road": 2300,
            "Dirt Road": -1800,
        }
    ).to_numpy()

    price_per_sqft = (
        3500
        + zoning_effect
        + topography_effect
        + soil_effect
        + access_effect
        + fsi_far * 1900
        + road_width * 45
        + frontage * 25
        + corner * 2600
        + utilities * 2400
        + micro_score * 1700
        - distance_to_road * 1.2
        + (latitude - 12.95) * 4500
        + (longitude - 77.65) * 3200
        + rng.normal(0, 1800, rows)
    ).clip(500)

    return pd.DataFrame(
        {
            "plot_area_sqft": plot_area,
            "frontage_width_ft": frontage.round(1),
            "topography_type": topography,
            "soil_type": soil,
            "corner_plot_flag": corner,
            "zoning_category": zoning,
            "fsi_far_ratio": fsi_far.round(2),
            "road_width_ft": road_width.round(1),
            "access_road_type": access_road,
            "distance_to_main_road_m": distance_to_road.round(1),
            "water_power_utility_access": utilities,
            "lat": latitude.round(6),
            "long": longitude.round(6),
            "micro_location_score": micro_score.round(1),
            LAND_TARGET: price_per_sqft.round(2),
        }
    )


def prepare_land_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(LAND_FEATURES + [LAND_TARGET]) - set(data.columns))
    if missing:
        raise ValueError(
            "Land training data is missing required columns: " + ", ".join(missing)
        )

    features = data[LAND_FEATURES].copy()
    for column in LAND_NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    target = pd.to_numeric(data[LAND_TARGET], errors="coerce")
    valid_rows = (
        target.notna()
        & np.isfinite(target)
        & (target > 0)
        & features["plot_area_sqft"].gt(0)
    )
    features = features.loc[valid_rows].reset_index(drop=True)
    target = target.loc[valid_rows].reset_index(drop=True)
    if len(features) < 30:
        raise ValueError(
            "At least 30 valid land records with positive plot area and price "
            "per square foot are required."
        )
    return features, target


def train_land_model(
    data: pd.DataFrame,
    *,
    model_path: str | Path = LAND_MODEL_PATH,
    target_currency: str = "INR",
    chart_path: str | Path | None = None,
    synthetic_data: bool = False,
) -> dict:
    if target_currency not in {"INR", "USD"}:
        raise ValueError("Target currency must be INR or USD.")
    features, target = prepare_land_data(data)
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, LAND_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, LAND_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "regressor__n_estimators": [120, 240],
            "regressor__max_depth": [None, 16],
            "regressor__min_samples_leaf": [1, 3],
        },
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(x_test)
    regressor = best_pipeline.named_steps["regressor"]
    transformed_names = best_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {"feature": transformed_names, "importance": regressor.feature_importances_}
    ).sort_values("importance", ascending=False)
    absolute_errors = np.abs(y_test.to_numpy() - predictions)
    bundle = {
        "model": best_pipeline,
        "feature_columns": LAND_FEATURES,
        "target": LAND_TARGET,
        "target_currency": target_currency,
        "synthetic_data": synthetic_data,
        "schema_version": LAND_MODEL_SCHEMA_VERSION,
        "training_rows": len(features),
        "best_params": search.best_params_,
        "metrics": {
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        },
        "interval_error": float(
            np.quantile(absolute_errors, 0.9, method="higher")
        ),
        "feature_importance": feature_importance.to_dict(orient="records"),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    if chart_path is not None:
        figure = px.bar(
            feature_importance.head(20).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Land model feature importance",
            labels={"importance": "Random forest importance", "feature": "Feature"},
        )
        figure.write_html(str(chart_path), include_plotlyjs="cdn")

    return bundle


def load_land_model(
    model_path: str | Path = LAND_MODEL_PATH,
    *,
    allow_synthetic: bool = False,
) -> dict:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Dedicated land model not found: {model_path}. "
            "Train it with `python train_land_model.py --data <land-data.csv> "
            "--target-currency INR`."
        )
    bundle = joblib.load(model_path)
    required_keys = {
        "model",
        "feature_columns",
        "target_currency",
        "synthetic_data",
        "schema_version",
    }
    if (
        not isinstance(bundle, dict)
        or not required_keys.issubset(bundle)
        or bundle["schema_version"] != LAND_MODEL_SCHEMA_VERSION
        or bundle["feature_columns"] != LAND_FEATURES
        or bundle["target_currency"] not in {"INR", "USD"}
        or not isinstance(bundle["synthetic_data"], bool)
    ):
        raise ValueError(
            f"Land model at {model_path} has an unsupported or invalid schema. "
            "Retrain it with the current train_land_model.py."
        )
    if bundle.get("synthetic_data", False) and not allow_synthetic:
        raise ValueError(
            "This land model was trained on synthetic demo records and is disabled "
            "for property valuations. Train with a labeled land-sales CSV."
        )
    return bundle


def explain_land_prediction(bundle: dict, property_features: pd.DataFrame):
    """Return TreeSHAP contributions for price-per-square-foot prediction."""
    import shap

    pipeline = bundle["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    transformed = preprocessor.transform(property_features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    try:
        explanation = shap.TreeExplainer(regressor)(transformed)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"Could not calculate land-model TreeSHAP values: {error}") from error
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]
    else:
        raise RuntimeError("TreeSHAP returned an unexpected explanation shape.")
    if len(values) != len(feature_names):
        raise RuntimeError("TreeSHAP values do not match the transformed features.")
    return sorted(
        zip(feature_names.astype(str), values.astype(float)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV containing labeled land records")
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="train a demo model on synthetic data (not suitable for real valuations)",
    )
    parser.add_argument("--synthetic-rows", type=int, default=600)
    parser.add_argument("--output", type=Path, default=LAND_MODEL_PATH)
    parser.add_argument(
        "--target-currency",
        choices=["INR", "USD"],
        default="INR",
        help="currency used by land_price_per_sqft in the training dataset",
    )
    parser.add_argument(
        "--importance-chart",
        type=Path,
        default=BASE_DIR / "land_feature_importance.html",
        help="path for the generated interactive feature-importance chart",
    )
    args = parser.parse_args()
    if args.generate_synthetic == bool(args.data):
        parser.error("Provide exactly one of --data or --generate-synthetic.")
    training_data = (
        make_synthetic_land_data(args.synthetic_rows)
        if args.generate_synthetic
        else pd.read_csv(args.data)
    )
    bundle = train_land_model(
        training_data,
        model_path=args.output,
        target_currency=args.target_currency,
        chart_path=args.importance_chart,
        synthetic_data=args.generate_synthetic,
    )
    print(f"Land model saved to: {args.output}")
    print(f"Training records: {bundle['training_rows']}")
    print(f"Target currency: {bundle['target_currency']} per square foot")
    print(f"Best parameters: {bundle['best_params']}")
    print(f"Holdout R²: {bundle['metrics']['r2']:.4f}")
    print(f"Holdout MAE: {bundle['metrics']['mae']:,.2f}")
    print(f"Holdout RMSE: {bundle['metrics']['rmse']:,.2f}")
    print(f"Feature importance chart: {args.importance_chart}")
    if args.generate_synthetic:
        print("WARNING: this synthetic-data model is for pipeline demos only.")


if __name__ == "__main__":
    main()
