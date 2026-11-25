
import streamlit as st
import pandas as pd
import os
import base64
import json
import math
import numpy as np
from datetime import datetime, timezone
from io import BytesIO
from html import escape
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageOps
from geopy.distance import geodesic
from price_model import MODEL_PATH, VISUAL_COLUMNS, load_or_train_model
from inspection_store import DEFAULT_INSPECTIONS_DIR, list_inspections, save_inspection
from train_land_model import (
    LAND_FEATURES,
    LAND_MODEL_PATH,
    explain_land_prediction,
    load_land_model,
    make_synthetic_land_data,
    prepare_land_data,
    train_land_model,
)
from ai_features import generate_property_listing, renovation_cost_range

# GPS and location search
from geopy.geocoders import Nominatim
from proptech_features import (
    calculate_investment_metrics,
    calculate_micro_location_score,
    explain_xgboost_prediction,
    fetch_nearby_amenities,
    fetch_usd_inr_rate,
    format_indian_rupees,
    generate_valuation_pdf,
    generate_valuation_summary_pdf,
    project_investment_cash_flows,
)
from external_property_service import (
    PropertyListingServiceError,
    fetch_live_property_listings,
    is_configured as live_property_api_is_configured,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Estate | Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
browser_geolocation = components.declare_component(
    "ai_estate_browser_geolocation",
    path=os.path.join(BASE_DIR, "components", "geolocation"),
)

csv_path = os.path.join(BASE_DIR, "properties.csv")
land_csv_path = os.path.join(BASE_DIR, "sample_land_schema.csv")
model_path = MODEL_PATH
images_folder = os.path.join(BASE_DIR, "images")
background_path = os.path.join(images_folder, "background.jpg")


# ============================================================
# LOAD BACKGROUND IMAGE
# ============================================================

def get_background_image(path):

    if not os.path.isfile(path):
        return None

    with open(path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    return f"data:image/jpeg;base64,{encoded_image}"


background_image = get_background_image(background_path)


# ============================================================
# LOAD DATASET
# ============================================================

try:
    source_data = pd.read_csv(csv_path)

except FileNotFoundError:
    st.error(f"Dataset file was not found: {csv_path}")
    st.stop()

except Exception as e:
    st.error(f"Could not load properties.csv: {e}")
    st.stop()


# ============================================================
# LOAD OR TRAIN VALUATION MODEL
# ============================================================

try:
    model_bundle = load_or_train_model(source_data, images_folder, model_path)
    model = model_bundle["model"]
    model_features = model_bundle["feature_columns"]
except Exception as e:
    st.error(f"Could not load or train the valuation model: {e}")
    st.stop()

required_form_features = {
    "sqft_living", "bedrooms", "bathrooms", "house_age", "lat", "long"
}
missing_form_features = sorted(required_form_features.difference(model_features))
if missing_form_features:
    st.error(
        "The loaded model does not include the dataset fields used by the "
        f"property form: {', '.join(missing_form_features)}. "
        "Run `python price_model.py` to retrain the model."
    )
    st.stop()

data = source_data.copy()
for column in data.columns:
    if column in {"id", "price", "area_name"}:
        continue
    data[column] = pd.to_numeric(data[column], errors="coerce")
if "location_score" not in data:
    if "zip_wealth_rank" in data:
        data["location_score"] = data["zip_wealth_rank"].rank(pct=True) * 10
    else:
        data["location_score"] = 5.0
for column in model_features:
    if column not in data:
        data[column] = 0.0

data[model_features] = data[model_features].replace(
    [float("inf"), -float("inf")],
    pd.NA
)
data = data.dropna(subset=model_features).reset_index(drop=True)
if data.empty:
    st.error("The dataset has no rows with valid values for the model features.")
    st.stop()

data["_row_key"] = range(len(data))


def load_or_generate_land_dataset(path):
    if os.path.isfile(path):
        land_data = pd.read_csv(path)
    else:
        land_data = make_synthetic_land_data(rows=80)
        sample_locations = [
            ("Vidyanagar", "Hubballi", 15.3647, 75.1240),
            ("Koramangala", "Bengaluru", 12.9352, 77.6245),
            ("Kuvempunagar", "Mysuru", 12.2866, 76.6376),
            ("Kadri", "Mangaluru", 12.8810, 74.8420),
            ("Tilakwadi", "Belagavi", 15.8497, 74.4977),
            ("Saptapur", "Dharwad", 15.4589, 75.0078),
            ("Sedam Road", "Kalaburagi", 17.3297, 76.8343),
            ("Manipal", "Udupi", 13.3525, 74.7928),
        ]
        rng = np.random.default_rng(42)
        selected_locations = [
            sample_locations[index % len(sample_locations)]
            for index in range(len(land_data))
        ]
        land_data["area_name"] = [location[0] for location in selected_locations]
        land_data["city"] = [location[1] for location in selected_locations]
        land_data["lat"] = [
            round(location[2] + rng.uniform(-0.003, 0.003), 6)
            for location in selected_locations
        ]
        land_data["long"] = [
            round(location[3] + rng.uniform(-0.003, 0.003), 6)
            for location in selected_locations
        ]
        land_data["land_address"] = (
            land_data["area_name"] + ", " + land_data["city"] + ", Karnataka"
        )
        land_data.to_csv(path, index=False)

    column_aliases = {
        "topography": "topography_type",
        "far_fsi_ratio": "fsi_far_ratio",
        "corner_plot": "corner_plot_flag",
        "utilities_available": "water_power_utility_access",
        "location_score": "micro_location_score",
        "latitude": "lat",
        "longitude": "long",
    }
    for source, target in column_aliases.items():
        if target not in land_data and source in land_data:
            land_data[target] = land_data[source]

    required_features = set(LAND_FEATURES + ["land_price_per_sqft"])
    missing_columns = sorted(required_features.difference(land_data.columns))
    if missing_columns:
        raise ValueError(
            "Land dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if "area_name" not in land_data:
        land_data["area_name"] = land_data.get(
            "locality",
            pd.Series("Karnataka", index=land_data.index),
        )
    if "city" not in land_data:
        land_data["city"] = "Karnataka"
    if "land_address" not in land_data:
        land_data["land_address"] = (
            land_data["area_name"].astype(str)
            + ", "
            + land_data["city"].astype(str)
            + ", Karnataka"
        )
    land_data["location_score"] = pd.to_numeric(
        land_data["micro_location_score"],
        errors="coerce",
    ).fillna(5.0)
    for column in (
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
        "land_price_per_sqft",
    ):
        land_data[column] = pd.to_numeric(land_data[column], errors="coerce")
        if land_data[column].isna().any():
            land_data[column] = land_data[column].fillna(
                land_data[column].median()
            )
    for column in (
        "topography_type",
        "soil_type",
        "zoning_category",
        "access_road_type",
    ):
        land_data[column] = land_data[column].fillna("Unknown").astype(str)
    land_data = land_data.reset_index(drop=True)
    land_data["_row_key"] = range(len(land_data))
    prepare_land_data(land_data)
    return land_data


if "land_df" not in st.session_state:
    try:
        st.session_state["land_df"] = load_or_generate_land_dataset(land_csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load the land dataset: {error}")
        st.stop()

land_df = st.session_state["land_df"]
nearby_listing_request = st.query_params.get("nearby_listing")
if nearby_listing_request:
    try:
        requested_type, requested_row = nearby_listing_request.split("-", maxsplit=1)
        requested_row = int(requested_row)
        if requested_type == "land" and 0 <= requested_row < len(land_df):
            st.session_state["asset_type"] = "Land"
            st.session_state["_requested_nearby_listing"] = (
                "Land",
                requested_row,
            )
        elif requested_type == "home" and 0 <= requested_row < len(data):
            st.session_state["asset_type"] = "Home"
            st.session_state["_requested_nearby_listing"] = (
                "Home",
                requested_row,
            )
        st.query_params.pop("nearby_listing", None)
    except (TypeError, ValueError):
        st.query_params.pop("nearby_listing", None)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_nearby_amenities(latitude, longitude):
    return fetch_nearby_amenities(latitude, longitude)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_live_property_listings(
    latitude,
    longitude,
    city,
    radius_km,
    search_type,
):
    return fetch_live_property_listings(
        latitude=float(latitude),
        longitude=float(longitude),
        city=str(city),
        radius_km=float(radius_km),
        search_type=str(search_type),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_usd_inr_rate():
    return fetch_usd_inr_rate()


@st.cache_data(max_entries=128, show_spinner=False)
def cached_photo_quality(image_content):
    from vision_module import assess_photo_quality

    return assess_photo_quality(image_content)


def render_photo_quality_badge(image_content):
    try:
        rating = cached_photo_quality(image_content)
    except (OSError, ValueError) as error:
        st.warning(f"Could not rate this photo's image quality: {error}")
        return

    stars = rating["stars"]
    st.caption(
        f"Visual quality: {'★' * stars}{'☆' * (5 - stars)} "
        f"({rating['quality_score']:.0f}/100) · "
        "photo clarity and lighting, not property condition"
    )


def update_micro_location_score(coordinates):
    try:
        nearby_results = cached_nearby_amenities(*coordinates)
        calculated_score = calculate_micro_location_score(nearby_results)
    except (RuntimeError, ValueError) as error:
        st.error(f"Could not calculate the micro-location score: {error}")
        return

    st.session_state.micro_location_lookup = {
        "coordinates": coordinates,
        "score": calculated_score,
        "results": nearby_results,
    }
    st.session_state.location_score_input = int(round(calculated_score))
    st.session_state["_location_score_coordinates"] = coordinates


def format_home_property_option(row_key):
    row = data.iloc[int(row_key)]
    dataset_id = row.get("id")
    area_name = row.get("area_name")
    label = f"Record {int(row_key) + 1}"
    if pd.notna(dataset_id):
        label = f"{label} • {normalize_property_id(dataset_id)}"
    if pd.notna(area_name) and str(area_name).strip():
        label = f"{label} • {area_name}"
    return label


def select_nearby_listing(asset_type, row_key):
    st.session_state.pop("_selected_live_listing", None)
    if asset_type == "Land":
        land_row = land_df.iloc[int(row_key)]
        st.session_state["asset_type"] = "Land"
        st.session_state["land_property_record"] = (
            f"Record {int(row_key) + 1} • "
            f"{land_row['area_name']}, {land_row['city']}"
        )
    else:
        st.session_state["asset_type"] = "Home"
        st.session_state["home_property_record"] = format_home_property_option(
            int(row_key)
        )


def clear_live_listing_selection():
    st.session_state.pop("_selected_live_listing", None)
    if st.session_state.get("asset_type") == "Home":
        selected_record = st.session_state.get("home_property_record")
        matching_rows = [
            row_key
            for row_key in data["_row_key"]
            if format_home_property_option(row_key) == selected_record
        ]
        if matching_rows:
            selected = data.iloc[int(matching_rows[0])]
            st.session_state["home_area_input"] = float(selected["sqft_living"])
            st.session_state["home_bedrooms_input"] = int(
                round(float(selected["bedrooms"]))
            )
            st.session_state["home_bathrooms_input"] = float(
                selected["bathrooms"]
            )
            st.session_state["home_age_input"] = int(
                round(float(selected["house_age"]))
            )


def select_live_listing(listing):
    st.session_state["_selected_live_listing"] = dict(listing)
    st.session_state["asset_type"] = "Home"
    st.session_state["home_area_input"] = float(listing.get("square_feet") or 0)
    st.session_state["home_bedrooms_input"] = int(
        round(float(listing.get("bedrooms") or 0))
    )
    st.session_state["home_bathrooms_input"] = float(
        listing.get("bathrooms") or 0
    )
    st.session_state["home_age_input"] = 0
    st.session_state.selected_lat = float(listing["latitude"])
    st.session_state.selected_lon = float(listing["longitude"])
    st.session_state.location_query = str(
        listing.get("address") or listing.get("title") or "External listing"
    )
    st.session_state.selected_location_label = st.session_state.location_query
    st.session_state.selected_location_source = (
        f"External listing · {listing.get('source', 'marketplace')}"
    )


def nearby_property_records(
    homes,
    lands,
    latitude,
    longitude,
    exchange_rate,
):
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("The searched location must have valid coordinates.") from error
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        raise ValueError("The searched location must have valid coordinates.")

    listings = []
    for row_key, row in homes.iterrows():
        price = pd.to_numeric(row.get("price"), errors="coerce")
        living_area = pd.to_numeric(row.get("sqft_living"), errors="coerce")
        plot_area = pd.to_numeric(row.get("sqft_lot"), errors="coerce")
        row_lat = pd.to_numeric(row.get("lat"), errors="coerce")
        row_lon = pd.to_numeric(row.get("long"), errors="coerce")
        if not all(
            pd.notna(value) and math.isfinite(float(value))
            for value in (price, living_area, plot_area, row_lat, row_lon)
        ):
            continue
        if (
            price <= 0
            or living_area <= 0
            or plot_area <= 0
            or not -90 <= row_lat <= 90
            or not -180 <= row_lon <= 180
        ):
            continue
        listings.append(
            {
                "asset_type": "Home",
                "row_key": int(row_key),
                "record": format_home_property_option(row_key),
                "property_id": normalize_property_id(row.get("id"))
                if pd.notna(row.get("id"))
                else f"Home record {int(row_key) + 1}",
                "area_name": str(row.get("area_name", "Not available")),
                "latitude": float(row_lat),
                "longitude": float(row_lon),
                "plot_area_sqft": float(plot_area),
                "comparison_area_sqft": float(living_area),
                "comparison_area_kind": "Living area",
                "frontage_width_ft": None,
                "price_total_inr": (
                    float(price) * exchange_rate
                    if exchange_rate is not None
                    else None
                ),
                "price_per_sqft_inr": (
                    float(price) * exchange_rate / float(living_area)
                    if exchange_rate is not None and living_area > 0
                    else None
                ),
                "source_currency": "USD",
            }
        )

    for row_key, row in lands.iterrows():
        plot_area = pd.to_numeric(row.get("plot_area_sqft"), errors="coerce")
        price_per_sqft = pd.to_numeric(
            row.get("land_price_per_sqft"),
            errors="coerce",
        )
        row_lat = pd.to_numeric(row.get("lat"), errors="coerce")
        row_lon = pd.to_numeric(row.get("long"), errors="coerce")
        if not all(
            pd.notna(value) and math.isfinite(float(value))
            for value in (plot_area, price_per_sqft, row_lat, row_lon)
        ):
            continue
        if (
            plot_area <= 0
            or price_per_sqft <= 0
            or not -90 <= row_lat <= 90
            or not -180 <= row_lon <= 180
        ):
            continue
        listings.append(
            {
                "asset_type": "Land",
                "row_key": int(row_key),
                "record": (
                    f"Record {int(row_key) + 1} • "
                    f"{row.get('area_name', 'Karnataka')}, "
                    f"{row.get('city', 'Karnataka')}"
                ),
                "property_id": f"Land record {int(row_key) + 1}",
                "area_name": str(row.get("land_address", row.get("area_name", ""))),
                "latitude": float(row_lat),
                "longitude": float(row_lon),
                "plot_area_sqft": float(plot_area),
                "comparison_area_sqft": float(plot_area),
                "comparison_area_kind": "Plot area",
                "frontage_width_ft": float(
                    pd.to_numeric(row.get("frontage_width_ft"), errors="coerce")
                )
                if pd.notna(
                    pd.to_numeric(row.get("frontage_width_ft"), errors="coerce")
                )
                else None,
                "price_total_inr": float(plot_area) * float(price_per_sqft),
                "price_per_sqft_inr": float(price_per_sqft),
                "source_currency": "INR",
            }
        )

    result = pd.DataFrame(listings)
    if result.empty:
        result["distance_km"] = pd.Series(dtype=float)
        return result

    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.radians(result["latitude"].to_numpy(dtype=float))
    lon2 = np.radians(result["longitude"].to_numpy(dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    central_angle = 2 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    result["distance_km"] = 6371.0088 * central_angle
    return result.sort_values("distance_km").reset_index(drop=True)


def make_nearby_listings_map(
    listings,
    latitude,
    longitude,
    location_name,
):
    nearby_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=12,
        control_scale=True,
    )
    folium.Marker(
        [latitude, longitude],
        tooltip="Searched location",
        popup=escape(location_name),
        icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
    ).add_to(nearby_map)

    buffer_layer = folium.FeatureGroup(name="Search radius buffers", show=True)
    for radius_m, color in ((500, "#10b981"), (1000, "#3b82f6"), (1500, "#f59e0b")):
        folium.Circle(
            [latitude, longitude],
            radius=radius_m,
            color=color,
            weight=1.5,
            fill=False,
            tooltip=f"{radius_m / 1000:g} km radius",
        ).add_to(buffer_layer)
    buffer_layer.add_to(nearby_map)

    valid_density = listings[
        listings["price_per_sqft_inr"].notna()
        & listings["price_per_sqft_inr"].gt(0)
    ]
    if not valid_density.empty:
        ppsf_values = valid_density["price_per_sqft_inr"].astype(float)
        normalization_limit = max(float(ppsf_values.quantile(0.95)), 1.0)
        heat_points = [
            [
                float(row["latitude"]),
                float(row["longitude"]),
                min(float(row["price_per_sqft_inr"]) / normalization_limit, 1.0),
            ]
            for row in valid_density.to_dict(orient="records")
        ]
        heat_layer = folium.FeatureGroup(
            name="Price per sq.ft density",
            show=False,
        )
        HeatMap(
            heat_points,
            radius=28,
            blur=22,
            min_opacity=0.25,
            name="Price per sq.ft density",
        ).add_to(heat_layer)
        heat_layer.add_to(nearby_map)

    comparable_layer = folium.FeatureGroup(
        name="Nearby comparables (color by distance)",
        show=True,
    )
    for listing in listings.to_dict(orient="records"):
        radius_link = f"?nearby_listing={listing['asset_type'].lower()}-{listing['row_key']}"
        price_label = (
            format_indian_rupees(listing["price_total_inr"])
            if listing["price_total_inr"] is not None
            else "INR conversion unavailable"
        )
        per_sqft_label = (
            format_indian_rupees(listing["price_per_sqft_inr"])
            if listing["price_per_sqft_inr"] is not None
            else "Not available"
        )
        popup = (
            f"<b>{escape(listing['asset_type'])} · "
            f"{escape(listing['property_id'])}</b>"
            f"<br>{escape(listing['area_name'])}"
            f"<br>Distance: {listing['distance_km']:.2f} km"
            f"<br>Plot area: {listing['plot_area_sqft']:,.0f} sq.ft"
            f"<br>Total price: {escape(price_label)}"
            f"<br>Price per sq.ft: {escape(per_sqft_label)}"
            f'<br><a href="{radius_link}" target="_self">'
            "Load this record for valuation</a>"
        )
        distance_color = (
            "green" if listing["distance_km"] < 0.5
            else "blue" if listing["distance_km"] < 1.0
            else "orange"
        )
        folium.Marker(
            [listing["latitude"], listing["longitude"]],
            tooltip=(
                f"{listing['asset_type']} · {listing['distance_km']:.2f} km · "
                f"{' <0.5 km' if listing['distance_km'] < 0.5 else '<1 km' if listing['distance_km'] < 1.0 else '≥1 km'}"
            ),
            popup=folium.Popup(popup, max_width=320),
            icon=folium.Icon(
                color=distance_color,
                icon="home",
                prefix="fa",
            ),
        ).add_to(comparable_layer)
    comparable_layer.add_to(nearby_map)
    folium.LayerControl(collapsed=False).add_to(nearby_map)
    return nearby_map


def nearest_historical_comparables(
    property_data,
    latitude,
    longitude,
    living_area,
    bedrooms,
    bathrooms,
    age,
    excluded_property_id=None,
    limit=3,
    radius_m=2000,
):
    feature_columns = ["sqft_living", "bedrooms", "bathrooms", "house_age"]
    required_columns = {
        "id",
        "price",
        "lat",
        "long",
        *feature_columns,
    }
    missing_columns = required_columns.difference(property_data.columns)
    if missing_columns:
        raise ValueError(
            "Historical comparables require dataset fields: "
            + ", ".join(sorted(missing_columns))
        )
    target_features = np.asarray(
        [living_area, bedrooms, bathrooms, age],
        dtype=float,
    )
    if not np.isfinite(target_features).all():
        raise ValueError("The current property's comparable-search features are invalid.")
    if (
        not math.isfinite(float(latitude))
        or not math.isfinite(float(longitude))
        or not -90 <= float(latitude) <= 90
        or not -180 <= float(longitude) <= 180
    ):
        raise ValueError("The searched location must have valid coordinates.")

    comparables = property_data[
        ["id", "price", *feature_columns, "lat", "long"]
        + (["year"] if "year" in property_data.columns else [])
    ].copy()
    for column in ("price", *feature_columns, "lat", "long"):
        comparables[column] = pd.to_numeric(
            comparables[column]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )
    comparables = comparables.dropna(
        subset=["price", *feature_columns, "lat", "long"]
    )
    comparables = comparables[
        comparables["lat"].between(-90, 90)
        & comparables["long"].between(-180, 180)
        & (comparables["price"] > 0)
        & (comparables["sqft_living"] > 0)
        & comparables["bedrooms"].ge(0)
        & comparables["bathrooms"].gt(0)
        & comparables["house_age"].ge(0)
    ].copy()
    if excluded_property_id is not None and pd.notna(excluded_property_id):
        excluded_id = normalize_property_id(excluded_property_id)
        comparables = comparables[
            comparables["id"].map(normalize_property_id) != excluded_id
        ]

    feature_scale = comparables[feature_columns].std(ddof=0).to_numpy(
        dtype=float,
        copy=True,
    )
    feature_scale[~np.isfinite(feature_scale) | (feature_scale == 0)] = 1.0
    comparables["distance_m"] = comparables.apply(
        lambda row: geodesic(
            (latitude, longitude),
            (row["lat"], row["long"]),
        ).meters,
        axis=1,
    )
    comparables = comparables[
        comparables["distance_m"].le(radius_m)
        & comparables["distance_m"].gt(1.0)
    ].copy()
    if comparables.empty:
        comparables["similarity_score"] = pd.Series(dtype=float)
        return comparables.reset_index(drop=True)

    feature_deltas = (
        comparables[feature_columns].to_numpy(dtype=float) - target_features
    ) / feature_scale
    north_delta_km = (
        comparables["lat"].to_numpy(dtype=float) - float(latitude)
    ) * 111.195
    east_delta_km = (
        (comparables["long"].to_numpy(dtype=float) - float(longitude))
        * 111.195
        * np.cos(
            np.radians(
                (comparables["lat"].to_numpy(dtype=float) + float(latitude)) / 2
            )
        )
    )
    combined_deltas = np.column_stack(
        [
            feature_deltas,
            north_delta_km / (radius_m / 1000),
            east_delta_km / (radius_m / 1000),
        ]
    )
    neighbors = NearestNeighbors(
        n_neighbors=min(limit, len(comparables)),
        metric="euclidean",
    )
    neighbors.fit(combined_deltas)
    euclidean_distances, nearest_indices = neighbors.kneighbors(
        np.zeros((1, combined_deltas.shape[1]))
    )
    comparables = comparables.iloc[nearest_indices[0]].copy()
    normalized_distances = euclidean_distances[0] / math.sqrt(
        combined_deltas.shape[1]
    )
    comparables["similarity_score"] = np.exp(-normalized_distances) * 100
    return comparables.reset_index(drop=True)


def make_comparables_map(comparables, latitude, longitude, exchange_rate):
    comps_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=13,
        control_scale=True,
    )
    folium.Marker(
        [latitude, longitude],
        tooltip="Selected location",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(comps_map)
    for index, comparable in comparables.iterrows():
        sale_year = (
            f"<br>Sale year: {int(comparable['year'])}"
            if "year" in comparable and pd.notna(comparable["year"])
            else ""
        )
        popup_html = (
            f"<b>Historical comp {index + 1}</b>{sale_year}"
            f"<br>Distance: {comparable['distance_m']:.0f} m"
            f"<br>Feature similarity: {comparable['similarity_score']:.0f}%"
            f"<br>Living area: {comparable['sqft_living']:,.0f} sq.ft"
            f"<br>Recorded price: "
            f"{escape(format_indian_rupees(comparable['price'] * exchange_rate))}"
        )
        folium.Marker(
            [comparable["lat"], comparable["long"]],
            tooltip=f"Comp {index + 1} · {comparable['distance_m']:.0f} m",
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(comps_map)
    folium.LayerControl(collapsed=True).add_to(comps_map)
    comps_map.fit_bounds(
        [[latitude, longitude]]
        + [
            [comparable["lat"], comparable["long"]]
            for _, comparable in comparables.iterrows()
        ],
        max_zoom=13,
    )
    return comps_map


def render_room_measurement(inspection_photos, dataset_identifier):
    with st.expander("Measure a room from a photo", expanded=False):
        st.warning(
            "A photo alone cannot reveal real-world dimensions. Calibrate against a "
            "known measurement in the same floor plane. Perspective can still affect accuracy."
        )
        st.caption(
            "For better estimates, photograph the floor as straight-on as possible, "
            "use a tape-measured reference, and measure rectangular rooms only."
        )

        photo_labels = [
            f"{index + 1}. {photo['filename']} ({photo['category']})"
            for index, photo in enumerate(inspection_photos)
        ]
        if not photo_labels:
            st.info("Add or select a property photo before measuring a room.")
            return

        photo_widget_key = f"measure_photo_{dataset_identifier}"
        selected_photo_label = st.selectbox(
            "Photo to measure",
            options=photo_labels,
            key=photo_widget_key,
        )
        if selected_photo_label is None or selected_photo_label not in photo_labels:
            st.warning("Please upload or select a valid room photo to start measuring.")
            return
        selected_photo_index = photo_labels.index(selected_photo_label)
        selected_photo = inspection_photos[selected_photo_index]
        measure_context = (
            f"{dataset_identifier}_{selected_photo_index}_"
            f"{selected_photo['filename']}"
        )

        try:
            measurement_image = ImageOps.exif_transpose(
                Image.open(BytesIO(selected_photo["content"]))
            ).convert("RGB")
        except Exception as error:
            st.error(f"Could not open the selected photo for measurement: {error}")
            return

        unit = st.selectbox(
            "Reference measurement unit",
            options=["feet", "meters"],
            key=f"measure_unit_{measure_context}",
        )
        reference_length = st.number_input(
            "Known reference length",
            min_value=0.01,
            value=3.0 if unit == "feet" else 1.0,
            step=0.25 if unit == "feet" else 0.1,
            help="For example, measure a floor edge, tile width, or another visible length with a tape.",
            key=f"measure_reference_{measure_context}_{unit}",
        )
        measurement_step = st.radio(
            "Measurement step",
            options=[
                "Calibrate scale",
                "Measure room length",
                "Measure room width",
            ],
            horizontal=True,
            key=f"measure_step_{measure_context}",
        )
        st.caption(
            "Drag across the known reference for calibration. Then select room length "
            "and width in turn and drag along each floor edge."
        )
        measurement_event = streamlit_image_coordinates(
            measurement_image,
            width="stretch",
            click_and_drag=True,
            cursor="crosshair",
            key=f"measure_image_{measure_context}_{measurement_step}",
        )

        measurement_key = f"_room_measure_{measure_context}"
        if measurement_key not in st.session_state:
            st.session_state[measurement_key] = {}
        if (
            isinstance(measurement_event, dict)
            and all(
                key in measurement_event
                for key in ("x1", "y1", "x2", "y2", "unix_time")
            )
        ):
            st.session_state[measurement_key][measurement_step] = measurement_event

        measure_state = st.session_state[measurement_key]

        def measured_pixels(event):
            if not isinstance(event, dict):
                return None
            delta_x = float(event["x2"]) - float(event["x1"])
            delta_y = float(event["y2"]) - float(event["y1"])
            distance = math.hypot(delta_x, delta_y)
            return distance if distance > 0 else None

        calibration_pixels = measured_pixels(
            measure_state.get("Calibrate scale")
        )
        if calibration_pixels:
            reference_feet = (
                reference_length if unit == "feet" else reference_length * 3.28084
            )
            feet_per_pixel = reference_feet / calibration_pixels
            measured_length_pixels = measured_pixels(
                measure_state.get("Measure room length")
            )
            measured_width_pixels = measured_pixels(
                measure_state.get("Measure room width")
            )

            st.caption(
                f"Calibrated scale: {reference_feet:.2f} ft over "
                f"{calibration_pixels:.0f} image pixels."
            )
            if measured_length_pixels and measured_width_pixels:
                room_length_ft = measured_length_pixels * feet_per_pixel
                room_width_ft = measured_width_pixels * feet_per_pixel
                room_area_sqft = room_length_ft * room_width_ft
                st.success(
                    f"Estimated room dimensions: {room_length_ft:.1f} ft × "
                    f"{room_width_ft:.1f} ft"
                )
                st.metric(
                    "Estimated room area",
                    f"{room_area_sqft:,.0f} sq.ft",
                    delta=f"{room_area_sqft * 0.092903:,.1f} sq.m",
                )
                st.caption(
                    "This is a single-photo, single-scale estimate—not a survey. "
                    "Camera perspective, uneven floor edges, and calibration error "
                    "can materially change the result. Room area is not the whole "
                    "property's built-up area and is not automatically used for valuation."
                )
            else:
                st.info(
                    "Scale is calibrated. Measure both the room length and width to "
                    "calculate approximate floor area."
                )
        else:
            st.info(
                "Start by selecting **Calibrate scale** and dragging across a visible "
                "reference whose real length you know."
            )


def render_feature_contributions(feature_contributions, exchange_rate, key):
    feature_labels = {
        "sqft_living": "Living area",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "house_age": "Property age",
        "location_score": "Location score (OSM / editable)",
        "lat": "Latitude",
        "long": "Longitude",
        "image_quality": "Photo clarity score",
        "image_brightness": "Photo brightness",
        "image_sharpness": "Photo sharpness",
        "image_finish_score": "Property finish & lighting score",
    }
    contribution_data = pd.DataFrame(
        [
            (
                feature_labels.get(name, name.replace("_", " ").title()),
                value * exchange_rate,
            )
            for name, value in feature_contributions
        ],
        columns=["Feature", "Contribution (INR)"],
    )
    if contribution_data.empty:
        return

    contribution_data["Direction"] = np.where(
        contribution_data["Contribution (INR)"] >= 0,
        "Raises estimate",
        "Lowers estimate",
    )
    contribution_data["Formatted contribution"] = contribution_data[
        "Contribution (INR)"
    ].map(
        lambda value: (
            f"{'+' if value >= 0 else '-'}"
            f"{format_indian_rupees(abs(value))}"
        )
    )
    figure = px.bar(
        contribution_data,
        x="Contribution (INR)",
        y="Feature",
        color="Direction",
        text="Formatted contribution",
        orientation="h",
        color_discrete_map={
            "Raises estimate": "#16803C",
            "Lowers estimate": "#C2413B",
        },
        hover_data={"Contribution (INR)": ":,.0f", "Direction": True},
    )
    figure.update_layout(
        xaxis_title="Contribution to estimate (INR)",
        yaxis_title=None,
        legend_title_text=None,
        xaxis_tickprefix="₹",
        xaxis_tickformat="~s",
        yaxis_autorange="reversed",
        margin={"l": 10, "r": 20, "t": 10, "b": 10},
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(figure, use_container_width=True, key=key)


def render_valuation_gauge(valuation, exchange_rate, key):
    """Render an INR valuation gauge with its model residual interval."""
    predicted = float(valuation["predicted_value"]) * exchange_rate
    lower = float(valuation["lower_bound"]) * exchange_rate
    upper = float(valuation["upper_bound"]) * exchange_rate
    axis_max = max(upper * 1.15, predicted * 1.2, 1.0)
    area_value = float(
        str(valuation["attributes"].get("Living area (sq.ft)", "0")).replace(",", "")
    )
    price_per_sqft = predicted / area_value if area_value > 0 else None
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted,
            number={"prefix": "₹", "valueformat": ",.0f", "font": {"size": 36}},
            title={"text": "Estimated property value (INR)"},
            gauge={
                "axis": {"range": [0, axis_max], "tickprefix": "₹", "tickformat": "~s"},
                "bar": {"color": "#2563eb", "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, lower], "color": "rgba(148,163,184,0.18)"},
                    {"range": [lower, upper], "color": "rgba(37,99,235,0.22)"},
                    {"range": [upper, axis_max], "color": "rgba(148,163,184,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#10b981", "width": 4},
                    "thickness": 0.8,
                    "value": predicted,
                },
            },
        )
    )
    figure.update_layout(
        height=290,
        margin={"l": 35, "r": 35, "t": 55, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#172033"},
    )
    st.plotly_chart(figure, use_container_width=True, key=key)
    lower_col, upper_col, sqft_col = st.columns(3)
    lower_col.metric("90% lower estimate", format_indian_rupees(lower))
    upper_col.metric("90% upper estimate", format_indian_rupees(upper))
    sqft_col.metric(
        "Estimated price / sq.ft",
        format_indian_rupees(price_per_sqft) if price_per_sqft is not None else "N/A",
    )
    st.caption(
        "The range is based on held-out model residuals; it is not a formal "
        "appraisal or a guarantee."
    )


def render_amenity_radar(amenities, key):
    radar_metrics = [
        ("Transit Access", "Public transit"),
        ("Schools & Education", "Schools"),
        ("Dining & Nightlife", "Dining & nightlife"),
        ("Parks & Green Space", "Parks & green space"),
        ("Safety / Walkability", "Safety / walkability"),
    ]
    metric_names = [name for name, _ in radar_metrics]
    scores = [
        float(amenities.get(category, {}).get("score", 0)) / 10
        for _, category in radar_metrics
    ]
    figure = go.Figure(
        go.Scatterpolar(
            r=scores + scores[:1],
            theta=metric_names + metric_names[:1],
            fill="toself",
            line={"color": "#2563eb", "width": 3},
            fillcolor="rgba(37,99,235,0.18)",
            hovertemplate="%{theta}: %{r:.1f}/10<extra></extra>",
        )
    )
    figure.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 10], "tick0": 0, "dtick": 2},
            "bgcolor": "rgba(255,255,255,0.15)",
        },
        showlegend=False,
        height=390,
        margin={"l": 55, "r": 55, "t": 25, "b": 25},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True, key=key)
    st.caption(
        "Scores scale the nearest mapped OSM feature's proximity to 1.5 km; "
        "they are coverage-dependent indicators, not safety ratings or service quality."
    )


def render_image_comparison(image_content, room_type, quality_score, photo_index):
    """Embed a local before/insight-reveal slider without transmitting the photo."""
    comparison_image = ImageOps.exif_transpose(
        Image.open(BytesIO(image_content))
    ).convert("RGB")
    comparison_image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    comparison_buffer = BytesIO()
    comparison_image.save(
        comparison_buffer,
        format="JPEG",
        quality=82,
        optimize=True,
    )
    encoded_image = base64.b64encode(comparison_buffer.getvalue()).decode("ascii")
    image_uri = f"data:image/jpeg;base64,{encoded_image}"
    room_label = escape(str(room_type or "Photo analysis unavailable"))
    quality_label = (
        f"{float(quality_score):.0f}/100"
        if quality_score is not None
        else "Unavailable"
    )
    component_html = f"""
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; }}
      .compare-shell {{ border: 1px solid #dbe5f0; border-radius: 16px; overflow: hidden;
        background: linear-gradient(135deg,#f8fbff,#eef6ff); padding: 14px; }}
      .compare-title {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:10px;
        font-size:12px; font-weight:700; color:#344054; }}
      .frame {{ position:relative; overflow:hidden; border-radius:12px; height:360px;
        background:#e5e7eb; }}
      .frame img {{ width:100%; height:100%; object-fit:contain; display:block; }}
      .reveal {{ position:absolute; inset:0; clip-path:inset(0 50% 0 0); }}
      .reveal::after {{ content:""; position:absolute; inset:0;
        background:linear-gradient(135deg,rgba(16,185,129,.12),transparent 55%); }}
      .tag {{ position:absolute; z-index:2; top:12px; left:12px; max-width:80%;
        padding:8px 10px; border:1px solid rgba(255,255,255,.6); border-radius:10px;
        color:#fff; background:rgba(15,23,42,.78); font-size:12px; line-height:1.45;
        backdrop-filter:blur(8px); }}
      input[type=range] {{ width:100%; margin:14px 0 0; accent-color:#2563eb; }}
      .note {{ color:#667085; font-size:11px; margin-top:8px; }}
      .open-modal {{ margin-top:8px; border:0; border-radius:9px; padding:8px 12px;
        color:#fff; background:#2563eb; cursor:pointer; font-weight:700; }}
      .modal {{ display:none; position:fixed; z-index:10; inset:0; padding:10px;
        background:rgba(15,23,42,.82); align-items:center; justify-content:center; }}
      .modal.open {{ display:flex; }}
      .modal-card {{ width:100%; padding:12px; border-radius:14px; background:#fff; }}
      .modal-head {{ display:flex; justify-content:space-between; align-items:center;
        font-weight:700; font-size:13px; margin-bottom:8px; }}
      .close-modal {{ border:0; border-radius:8px; padding:6px 10px;
        color:#172033; background:#e2e8f0; cursor:pointer; }}
      .modal .frame {{ height:300px; }}
    </style>
    <div class="compare-shell">
      <div class="compare-title"><span>Original</span><span>AI insight overlay · photo {photo_index + 1}</span></div>
      <div class="frame">
        <img src="{image_uri}" alt="Original property photo">
        <div class="reveal" id="reveal">
          <img src="{image_uri}" alt="Photo with AI insight labels">
          <div class="tag">Visual scene: {room_label}<br>Photo clarity: {quality_label}</div>
        </div>
      </div>
      <input id="reveal-control" type="range" min="0" max="100" value="50"
        aria-label="Reveal AI insight overlay">
      <div class="note">Drag to compare the original with photo-analysis labels. Labels summarize image cues; no room segmentation is implied.</div>
      <button class="open-modal" id="open-modal" type="button">Open comparison modal</button>
    </div>
    <div class="modal" id="modal" role="dialog" aria-modal="true" aria-label="Photo comparison">
      <div class="modal-card">
        <div class="modal-head"><span>Photo comparison · {photo_index + 1}</span>
          <button class="close-modal" id="close-modal" type="button">Close</button></div>
        <div class="frame">
          <img src="{image_uri}" alt="Original property photo">
          <div class="reveal" id="modal-reveal">
            <img src="{image_uri}" alt="AI insight overlay">
            <div class="tag">Visual scene: {room_label}<br>Photo clarity: {quality_label}</div>
          </div>
        </div>
        <input id="modal-control" type="range" min="0" max="100" value="50"
          aria-label="Reveal AI insight overlay in modal">
      </div>
    </div>
    <script>
      const pairControls = [
        [document.getElementById("reveal-control"), document.getElementById("reveal")],
        [document.getElementById("modal-control"), document.getElementById("modal-reveal")]
      ];
      pairControls.forEach(([control, reveal]) => control.addEventListener("input", () => {{
        reveal.style.clipPath = `inset(0 ${{100 - control.value}}% 0 0)`;
      }}));
      const modal = document.getElementById("modal");
      document.getElementById("open-modal").addEventListener("click", () => modal.classList.add("open"));
      document.getElementById("close-modal").addEventListener("click", () => modal.classList.remove("open"));
    </script>
    """
    components.html(component_html, height=520, scrolling=False)


def render_land_contributions(contributions, plot_area_sqft, target_currency, exchange_rate):
    if target_currency == "INR":
        currency_rate = 1.0
    elif target_currency == "USD":
        if exchange_rate is None:
            st.error("Land-model USD contributions need the USD-to-INR exchange rate.")
            return
        currency_rate = exchange_rate
    else:
        raise ValueError(f"Unsupported land-model target currency: {target_currency}")

    rows = [
        (
            name.replace("_", " ")
            .replace("numeric__", "")
            .replace("categorical__", "")
            .title(),
            value * plot_area_sqft * currency_rate,
        )
        for name, value in contributions[:12]
    ]
    chart_data = pd.DataFrame(
        rows,
        columns=["Feature", "Estimated value contribution (INR)"],
    )
    display_data = chart_data.copy()
    display_data["Estimated value contribution (INR)"] = (
        display_data["Estimated value contribution (INR)"].map(
            lambda value: (
                f"{'+' if value >= 0 else '-'}"
                f"{format_indian_rupees(abs(value))}"
            )
        )
    )
    st.bar_chart(
        chart_data,
        x="Feature",
        y="Estimated value contribution (INR)",
        horizontal=True,
    )
    st.table(display_data)


try:
    usd_inr_rate, exchange_rate_updated_at = cached_usd_inr_rate()
    exchange_rate_error = None
except RuntimeError as error:
    usd_inr_rate = None
    exchange_rate_updated_at = None
    exchange_rate_error = str(error)


# ============================================================
# PROPERTY IMAGE FUNCTION
# ============================================================

def normalize_property_id(property_id):
    identifier = str(property_id).strip()
    try:
        numeric_identifier = float(identifier)
    except ValueError:
        return identifier

    if numeric_identifier.is_integer():
        return str(int(numeric_identifier))
    return identifier


def get_property_images(property_id):

    if not os.path.isdir(images_folder):
        return []

    property_folder = os.path.join(
        images_folder,
        normalize_property_id(property_id)
    )

    if not os.path.isdir(property_folder):
        return []

    images = []

    try:
        file_names = os.listdir(property_folder)
    except OSError:
        return []

    for file_name in file_names:

        file_path = os.path.join(
            property_folder,
            file_name
        )

        if not os.path.isfile(file_path):
            continue

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            images.append(file_path)

    images.sort()

    return images


def get_location_details(address, fallback_area=None):
    raw_address = getattr(address, "raw", {}).get("address", {})
    if not isinstance(raw_address, dict):
        raw_address = {}

    area = next(
        (
            raw_address[key]
            for key in ("neighbourhood", "suburb", "city_district", "quarter", "residential")
            if raw_address.get(key)
        ),
        "",
    )
    city = next(
        (
            raw_address[key]
            for key in ("city", "town", "municipality", "village", "county")
            if raw_address.get(key)
        ),
        "",
    )

    if not area and fallback_area:
        parts = [part.strip() for part in str(fallback_area).split(",", maxsplit=1)]
        area = parts[0]
        if not city and len(parts) > 1:
            city = parts[1]

    return {
        "area": area,
        "city": city,
        "road": next(
            (
                raw_address[key]
                for key in ("road", "pedestrian", "footway", "street")
                if raw_address.get(key)
            ),
            "",
        ),
        "state": raw_address.get("state", ""),
        "postcode": raw_address.get("postcode", ""),
        "country": raw_address.get("country", ""),
    }


valuation_ready_ids = {
    normalize_property_id(property_id)
    for property_id in data["id"]
    if pd.notna(property_id)
}
image_only_property_ids = sorted({
    normalize_property_id(property_id)
    for property_id in source_data["id"]
    if pd.notna(property_id)
    and normalize_property_id(property_id) not in valuation_ready_ids
    and get_property_images(property_id)
})


# ============================================================
# PROFESSIONAL CSS
# ============================================================

dark_glass_theme = st.toggle(
    "Dark glass theme",
    value=False,
    key="dark_glass_theme",
)

dark_theme_css = ""
if dark_glass_theme:
    dark_theme_css = """
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(37,99,235,.2), transparent 30%),
            radial-gradient(circle at 95% 35%, rgba(16,185,129,.1), transparent 25%),
            #0b1220 !important;
    }
    [data-testid="stAppViewContainer"] {
        --text-color: #e5edf8;
        --secondary-text-color: #aab8cb;
        color: #e5edf8 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricDelta"],
    [data-testid="stAppViewContainer"] [data-testid="stTable"],
    [data-testid="stAppViewContainer"] [data-testid="stDataFrame"],
    [data-testid="stAppViewContainer"] [data-testid="stAlert"] {
        color: #e5edf8 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMetricDelta"] *,
    [data-testid="stAppViewContainer"] [data-testid="stAlert"] p {
        color: #e5edf8 !important;
        -webkit-text-fill-color: #e5edf8 !important;
    }
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
    }
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
        -webkit-text-fill-color: #60a5fa !important;
    }
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea,
    [data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
        color: #e5edf8 !important;
        background-color: rgba(15,23,42,.82) !important;
        border-color: rgba(148,163,184,.28) !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(11,18,32,.78) !important;
    }
    .topbar, .hero-copy, .metric-card, div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15,23,42,.68) !important;
        border-color: rgba(148,163,184,.24) !important;
    }
    .main-title {
        background-image: linear-gradient(135deg,#f8fafc 8%,#60a5fa 55%,#6ee7b7 100%) !important;
    }
    .subtitle, .metric-value, .metric-label, .summary-value, .summary-label {
        color: #e5edf8 !important;
    }
    .top-links { color: #cbd5e1 !important; }
    """

background_css = ""

if background_image:

    background_css = f"""
    background-image:
        radial-gradient(circle at top left, rgba(20, 121, 232, 0.14), transparent 28%),
        radial-gradient(circle at right center, rgba(14, 165, 233, 0.12), transparent 30%),
        linear-gradient(
            rgba(245, 248, 252, 0.86),
            rgba(245, 248, 252, 0.96)
        ),
        url("{background_image}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    """

else:

    background_css = """
    background:
        radial-gradient(circle at top left, rgba(20, 121, 232, 0.12), transparent 28%),
        radial-gradient(circle at right center, rgba(14, 165, 233, 0.10), transparent 25%),
        #F7F9FC;
    """


st.markdown(
    f"""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
    }}

    .stApp {{
        {background_css}
    }}

    [data-testid="stAppViewContainer"] {{
        --text-color: #101828;
        --secondary-text-color: #475467;
        color: #101828 !important;
    }}

    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppVīiewContainer"] [data-testid="stMetricLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
    [data-testid="stAppViewContainer"] [data-testid="stMetricDelta"],
    [data-testid="stAppViewContainer"] [data-testid="stTable"],
    [data-testid="stAppViewContainer"] [data-testid="stDataFrame"],
    [data-testid="stAppViewContainer"] [data-testid="stRadio"],
    [data-testid="stAppViewContainer"] [data-testid="stSlider"],
    [data-testid="stAppViewContainer"] [data-testid="stAlert"] {{
        color: #101828 !important;
    }}

    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMetricValue"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMetricDelta"] *,
    [data-testid="stAppViewContainer"] [data-testid="stTable"] *,
    [data-testid="stAppViewContainer"] [data-testid="stDataFrame"] *,
    [data-testid="stAppViewContainer"] [data-testid="stRadio"] label,
    [data-testid="stAppViewContainer"] [data-testid="stSlider"] label,
    [data-testid="stAppViewContainer"] [data-testid="stAlert"] p {{
        color: #101828 !important;
        -webkit-text-fill-color: #101828 !important;
    }}

    [data-testid="stAppViewContainer"] button[data-baseweb="tab"] {{
        color: #344054 !important;
        -webkit-text-fill-color: #344054 !important;
    }}

    [data-testid="stAppViewContainer"] button[data-baseweb="tab"][aria-selected="true"] {{
        color: #0B5BC2 !important;
        -webkit-text-fill-color: #0B5BC2 !important;
    }}

    [data-testid="stAppViewContainer"] .stButton > button,
    [data-testid="stAppViewContainer"] .stButton > button * {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}

    * {{
        scrollbar-width: thin;
        scrollbar-color: rgba(148, 163, 184, 0.7) transparent;
    }}

    ::selection {{
        background: rgba(20, 121, 232, 0.18);
    }}

    header[data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .block-container {{
        max-width: 1280px;
        padding-top: 32px;
        padding-bottom: 56px;
    }}

    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        padding: 14px 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
    }}

    .brand-mark {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.6px;
        color: #0F66D0;
        text-transform: uppercase;
    }}

    .brand-badge {{
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: linear-gradient(135deg, #1479E8 0%, #5aa9ff 100%);
        color: #fff;
        box-shadow: 0 8px 18px rgba(20, 121, 232, 0.18);
        font-size: 15px;
    }}

    .top-links {{
        display: flex;
        align-items: center;
        gap: 18px;
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }}

    .top-links span {{
        opacity: 0.9;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.10);
        color: #15803d;
        border: 1px solid rgba(34, 197, 94, 0.18);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        display: inline-block;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.12);
    }}

    .brand {{
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #0F66D0;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}

    .main-title {{
        font-size: clamp(2.5rem, 4vw, 4.2rem);
        font-weight: 800;
        color: #101828;
        letter-spacing: -2.4px;
        line-height: 1.04;
        margin-bottom: 8px;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.4);
        background: linear-gradient(135deg, #0f172a 8%, #0f66d0 48%, #60a5fa 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Plus Jakarta Sans', 'Manrope', sans-serif;
    }}

    .subtitle {{
        color: #334155;
        font-size: 16px;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 28px;
        font-weight: 600;
    }}

    .badge-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 22px;
    }}

    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 7px 12px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(20, 121, 232, 0.10), rgba(69, 182, 255, 0.12));
        color: #0F66D0;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid rgba(20, 121, 232, 0.12);
    }}

    .hero-shell {{
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.75fr);
        gap: 22px;
        align-items: stretch;
        margin-bottom: 18px;
    }}

    .hero-copy {{
        background: rgba(255, 255, 255, 0.26);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 26px;
        padding: 24px 22px 20px 22px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.04);
        backdrop-filter: blur(10px);
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-top: 10px;
        margin-bottom: 0;
        max-width: 780px;
    }}

    .metric-card {{
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
        padding: 18px 18px 16px 18px;
        backdrop-filter: blur(8px);
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        animation: metric-rise 550ms ease both;
    }}

    .metric-card:hover, .showcase-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(37, 99, 235, 0.32);
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.13);
    }}

    @keyframes metric-rise {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .metric-label {{
        display: block;
        font-size: 11px;
        color: #667085;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 12px;
    }}

    .metric-value {{
        display: block;
        font-size: clamp(1.4rem, 2vw, 2rem);
        font-weight: 800;
        color: #101828;
        letter-spacing: -1px;
        line-height: 1.2;
    }}

    .metric-trend {{
        display: inline-block;
        margin-top: 10px;
        font-size: 11px;
        font-weight: 700;
        color: #15803d;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.12);
        border-radius: 999px;
        padding: 5px 8px;
    }}

    .showcase-card {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(14, 116, 144, 0.90));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 26px;
        padding: 22px 20px;
        box-shadow: 0 22px 43px rgba(15, 23, 42, 0.18);
        color: #F8FAFC;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 220px;
        transition: transform 220ms ease, box-shadow 220ms ease;
        animation: metric-rise 650ms ease both;
    }}

    .showcase-card .mini-label {{
        font-size: 11px;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: rgba(191, 219, 254, 0.8);
        font-weight: 700;
    }}

    .showcase-card .mini-title {{
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1.4px;
        margin-top: 18px;
        color: #FFFFFF;
        font-family: 'Plus Jakarta Sans', 'Manrope', sans-serif;
    }}

    .showcase-card .mini-copy {{
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.6;
        color: rgba(226, 232, 240, 0.82);
    }}

    .mini-stats {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-top: 18px;
        padding-top: 16px;
        border-top: 1px solid rgba(148, 163, 184, 0.22);
    }}

    .mini-box {{
        flex: 1;
    }}

    .mini-box span {{
        display: block;
    }}

    .mini-box .mini-number {{
        font-size: 20px;
        font-weight: 800;
        color: #fff;
    }}

    .mini-box .mini-text {{
        font-size: 11px;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        color: rgba(191, 219, 254, 0.75);
        margin-top: 4px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(15, 23, 42, 0.06) !important;
        border-radius: 18px !important;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08) !important;
        padding: 18px !important;
        backdrop-filter: blur(8px);
    }}

    h1, h2, h3 {{
        color: #101828 !important;
        font-weight: 700 !important;
    }}

    h2 {{
        font-size: 20px !important;
    }}

    h3 {{
        font-size: 17px !important;
    }}

    .section-title {{
        font-size: 18px;
        font-weight: 700;
        color: #101828;
        margin-bottom: 6px;
    }}

    .section-description {{
        font-size: 13px;
        color: #475467;
        margin-bottom: 14px;
        font-weight: 600;
    }}

    label {{
        color: #344054 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    div[data-baseweb="input"] {{
        background: rgba(255, 255, 255, 0.98) !important;
        border: 1px solid #D0D5DD !important;
        border-radius: 10px !important;
        min-height: 44px !important;
        box-shadow: inset 0 1px 2px rgba(16, 24, 40, 0.02);
    }}

    div[data-baseweb="input"]:focus-within {{
        border: 1px solid #1479E8 !important;
        box-shadow: 0 0 0 3px rgba(20, 121, 232, 0.12) !important;
    }}

    div[data-baseweb="input"] input {{
        color: #101828 !important;
        background: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }}

    [data-testid="stTextInput"] > div {{
        background: #FFFFFF !important;
        border-radius: 10px !important;
    }}

    [data-testid="stTextInput"] input {{
        background: #FFFFFF !important;
        color: #101828 !important;
        -webkit-text-fill-color: #101828 !important;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: #667085 !important;
    }}

    div[data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.3);
    }}

    button[data-baseweb="tab"] {{
        border-radius: 12px 12px 0 0;
        color: #475467;
        font-weight: 700;
        padding: 10px 14px;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #0B5BC2;
        background: rgba(20, 121, 232, 0.08);
    }}

    [data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 14px;
    }}

    [data-testid="stTextInput"] > div:focus-within {{
        border: 1px solid #1479E8 !important;
        box-shadow: 0 0 0 3px rgba(20, 121, 232, 0.12) !important;
    }}

    div[data-baseweb="select"] > div {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid #D0D5DD !important;
        border-radius: 10px !important;
        min-height: 44px !important;
    }}

    div[data-baseweb="select"] span {{
        color: #101828 !important;
        font-family: 'Inter', sans-serif !important;
    }}

    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #1479E8 0%, #0B5BC2 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 48px !important;
        font-family: 'Plus Jakarta Sans', 'Manrope', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        box-shadow: 0 12px 24px rgba(20, 121, 232, 0.18) !important;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #0F66D0 0%, #0A4FAF 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(20, 121, 232, 0.24) !important;
    }}

    [data-testid="stFileUploader"] section {{
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(208, 213, 221, 0.9);
        border-radius: 12px;
    }}

    [data-testid="stImage"] img {{
        border-radius: 12px !important;
        object-fit: cover !important;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }}

    div[data-testid="stAlert"] {{
        border-radius: 12px !important;
        font-size: 13px;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }}

    .valuation-card {{
        background: linear-gradient(135deg, rgba(240, 247, 255, 0.98), rgba(255, 255, 255, 1));
        border: 1px solid rgba(125, 211, 252, 0.35);
        border-radius: 18px;
        padding: 22px;
        margin-top: 12px;
        margin-bottom: 15px;
        box-shadow: 0 12px 26px rgba(14, 116, 144, 0.08);
    }}

    .valuation-label {{
        font-size: 12px;
        font-weight: 700;
        color: #475467;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    .valuation-price {{
        font-size: 32px;
        font-weight: 800;
        color: #1479E8;
        margin-top: 6px;
    }}

    .valuation-note {{
        font-size: 12px;
        color: #475467;
        margin-top: 4px;
        font-weight: 600;
    }}

    .summary-item {{
        padding: 14px 0;
        border-bottom: 1px solid rgba(234, 236, 240, 0.85);
    }}

    .summary-label {{
        font-size: 12px;
        color: #475467;
        margin-bottom: 3px;
        font-weight: 700;
    }}

    .summary-value {{
        font-size: 15px;
        color: #101828;
        font-weight: 600;
    }}

    .footer {{
        text-align: center;
        color: #475467;
        font-size: 12px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid rgba(234, 236, 240, 0.9);
        font-weight: 600;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 12px !important;
            padding-right: 12px !important;
            padding-top: 12px !important;
            max-width: 100% !important;
        }}

        .topbar {{
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 10px !important;
            padding: 12px 14px !important;
        }}

        .top-links {{
            display: none !important;
        }}

        .hero-shell {{
            grid-template-columns: 1fr !important;
            gap: 14px !important;
        }}

        .metric-grid {{
            grid-template-columns: 1fr !important;
            gap: 12px !important;
        }}

        .main-title {{
            font-size: 2.2rem !important;
            letter-spacing: -1.5px !important;
        }}

        .subtitle {{
            font-size: 14px !important;
        }}

        .stButton > button {{
            min-height: 52px !important;
            font-size: 15px !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 14px !important;
            border-radius: 16px !important;
        }}

        .stRadio > div {{
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }}

        .stTextInput input, .stNumberInput input {{
            font-size: 16px !important;
        }}
    }}

    {dark_theme_css}

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

if usd_inr_rate is not None:
    median_dataset_price = pd.to_numeric(
        source_data["price"],
        errors="coerce",
    ).median()
    median_dataset_price_label = (
        format_indian_rupees(float(median_dataset_price) * usd_inr_rate)
        if pd.notna(median_dataset_price)
        else "Not available"
    )
    exchange_rate_label = (
        f"1 USD = ₹{usd_inr_rate:,.2f} · updated {exchange_rate_updated_at}"
    )
else:
    median_dataset_price_label = "Not available"
    exchange_rate_label = "USD-to-INR rate unavailable"

dataset_demand_label = f"{len(source_data):,}"
dataset_growth_label = "N/A"
growth_period_label = "No dated trend"
if {"year", "price"}.issubset(source_data.columns):
    growth_data = source_data[["year", "price"]].copy()
    growth_data["year"] = pd.to_numeric(growth_data["year"], errors="coerce")
    growth_data["price"] = pd.to_numeric(growth_data["price"], errors="coerce")
    yearly_medians = (
        growth_data.dropna(subset=["year", "price"])
        .groupby("year")["price"]
        .median()
        .sort_index()
    )
    if len(yearly_medians) >= 2:
        first_year = float(yearly_medians.index[0])
        last_year = float(yearly_medians.index[-1])
        first_median = float(yearly_medians.iloc[0])
        last_median = float(yearly_medians.iloc[-1])
        if last_year > first_year and first_median > 0 and last_median > 0:
            annualized_growth = (
                (last_median / first_median) ** (1 / (last_year - first_year)) - 1
            ) * 100
            dataset_growth_label = f"{annualized_growth:+.1f}%"
            growth_period_label = (
                f"Dataset median CAGR · {first_year:.0f}–{last_year:.0f}"
            )

st.markdown(
    '''
    <div class="topbar">
        <div class="brand-mark">
            <span class="brand-badge">🏠</span>
            <span>AI Estate</span>
        </div>
        <div class="top-links">
            <span>Dashboard</span>
            <span>Valuation</span>
            <span>Insights</span>
        </div>
        <div class="status-pill"><span class="status-dot"></span>Live model</div>
    </div>
    ''',
    unsafe_allow_html=True
)
if exchange_rate_error:
    st.error(
        f"{exchange_rate_error} INR valuation display is unavailable until the "
        "exchange-rate service responds."
    )
else:
    st.caption(
        f"Latest published exchange rate: {exchange_rate_label} "
        "· Source: ExchangeRate-API"
    )

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-copy">
            <div class="brand">AI • REAL ESTATE INTELLIGENCE</div>
            <div class="main-title">Smart Property Valuation</div>
            <div class="subtitle">
                Estimate residential property value using machine learning and structured property information.
            </div>
            <div class="badge-container">
                <span class="badge">AI Powered</span>
                <span class="badge">Machine Learning</span>
                <span class="badge">Instant Valuation</span>
            </div>
            <div class="metric-grid">
                <div class="metric-card">
                    <span class="metric-label">Dataset listings</span>
                    <span class="metric-value">{len(source_data):,}</span>
                    <span class="metric-trend">Local property records</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Valuation interval</span>
                    <span class="metric-value">90%</span>
                    <span class="metric-trend">Held-out residual estimate</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Exchange rate</span>
                    <span class="metric-value">{f"₹{usd_inr_rate:,.2f}" if usd_inr_rate is not None else "Unavailable"}</span>
                    <span class="metric-trend">INR per USD</span>
                </div>
            </div>
        </div>
        <div class="showcase-card">
            <div>
                <div class="mini-label">Dataset median value</div>
                <div class="mini-title">{median_dataset_price_label}</div>
                <div class="mini-copy">Converted from the local dataset using the published USD-to-INR rate; not live market data.</div>
            </div>
            <div class="mini-stats">
                <div class="mini-box">
                    <span class="mini-number">{dataset_demand_label}</span>
                    <span class="mini-text">Demand · dataset records</span>
                </div>
                <div class="mini-box">
                    <span class="mini-number">{dataset_growth_label}</span>
                    <span class="mini-text">{growth_period_label}</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1.65, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    # PROPERTY DETAILS

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Property Information</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Choose the asset type and enter its details below.'
            '</div>',
            unsafe_allow_html=True
        )

        asset_type = st.radio(
            "Asset type",
            options=["Home", "Land"],
            horizontal=True,
            key="asset_type",
            on_change=clear_live_listing_selection,
        )

        if asset_type == "Home":
            property_options = [
                format_home_property_option(row_key)
                for row_key in data["_row_key"]
            ]
            requested_listing = st.session_state.pop(
                "_requested_nearby_listing",
                None,
            )
            if requested_listing and requested_listing[0] == "Home":
                st.session_state["home_property_record"] = property_options[
                    requested_listing[1]
                ]
            property_key = st.selectbox(
                "Property record",
                options=property_options,
                key="home_property_record",
                on_change=clear_live_listing_selection,
            )
            row_position = property_options.index(property_key)
            selected_property = data.iloc[row_position]
            source_identifier = selected_property.get("id")
            if pd.isna(source_identifier):
                dataset_identifier = f"row-{row_position + 1}"
            else:
                dataset_identifier = normalize_property_id(source_identifier)
            property_label = property_key
        else:
            land_property_options = [
                (
                    f"Record {row_key + 1} • "
                    f"{land_df.iloc[row_key]['area_name']}, "
                    f"{land_df.iloc[row_key]['city']}"
                )
                for row_key in land_df["_row_key"]
            ]
            requested_listing = st.session_state.pop(
                "_requested_nearby_listing",
                None,
            )
            if requested_listing and requested_listing[0] == "Land":
                st.session_state["land_property_record"] = (
                    land_property_options[requested_listing[1]]
                )
            property_key = st.selectbox(
                "Property record",
                options=land_property_options,
                key="land_property_record",
                on_change=clear_live_listing_selection,
            )
            row_position = land_property_options.index(property_key)
            selected_property = land_df.iloc[row_position]
            source_identifier = selected_property.get("id")
            if pd.isna(source_identifier):
                source_identifier = None
            dataset_identifier = f"land-row-{row_position + 1}"
            property_label = property_key

            if st.session_state.get("_land_form_record_key") != property_key:
                form_defaults = {
                    "land_plot_area": max(
                        1.0,
                        float(selected_property["plot_area_sqft"]),
                    ),
                    "land_frontage_width": float(
                        selected_property["frontage_width_ft"]
                    ),
                    "land_zoning": str(selected_property["zoning_category"]),
                    "land_topography": str(
                        selected_property["topography_type"]
                    ),
                    "land_soil_type": str(selected_property["soil_type"]),
                    "land_road_width": float(selected_property["road_width_ft"]),
                    "land_access_road_type": str(
                        selected_property["access_road_type"]
                    ),
                    "land_distance_to_main_road": float(
                        selected_property["distance_to_main_road_m"]
                    ),
                    "land_fsi_far_ratio": float(
                        selected_property["fsi_far_ratio"]
                    ),
                    "land_corner_plot": bool(
                        selected_property["corner_plot_flag"]
                    ),
                    "land_utilities_available": bool(
                        selected_property["water_power_utility_access"]
                    ),
                    "location_score_input": max(
                        2,
                        min(
                            10,
                            int(round(float(selected_property["location_score"]))),
                        ),
                    ),
                }
                st.session_state.update(form_defaults)
                st.session_state["_land_form_record_key"] = property_key
# ============================================================
# GPS LOCATION
# ============================================================

active_property_key = f"{asset_type}:{property_key}"
if (
    st.session_state.get("_active_property_key") != active_property_key
    or (
        asset_type == "Home"
        and "home_area_input" not in st.session_state
    )
):
    st.session_state["_active_property_key"] = active_property_key
    selected_live_listing = st.session_state.get("_selected_live_listing")
    if not selected_live_listing:
        st.session_state.selected_lat = float(selected_property["lat"])
        st.session_state.selected_lon = float(selected_property["long"])
        selected_area = selected_property.get("area_name")
        if pd.notna(selected_area) and str(selected_area).strip():
            city = str(selected_property.get("city", "")).strip()
            address = str(
                selected_property.get(
                    "land_address",
                    f"{selected_area}, {city}".strip(", "),
                )
            )
            st.session_state.location_query = address
            st.session_state.selected_location_label = address
            st.session_state.selected_location_source = (
                "Representative locality center"
            )
            st.session_state.selected_location_details = get_location_details(
                None, fallback_area=f"{selected_area}, {city}".strip(", ")
            )
        else:
            st.session_state.location_query = "Selected property location"
            st.session_state.selected_location_label = "Selected property location"
            st.session_state.selected_location_source = "Dataset coordinates"
            st.session_state.selected_location_details = get_location_details(None)
        if asset_type == "Home":
            st.session_state.home_area_input = float(selected_property["sqft_living"])
            st.session_state.home_bedrooms_input = int(
                round(float(selected_property["bedrooms"]))
            )
            st.session_state.home_bathrooms_input = float(
                selected_property["bathrooms"]
            )
            st.session_state.home_age_input = int(
                round(float(selected_property["house_age"]))
            )
    elif asset_type == "Home":
        st.session_state.home_area_input = float(
            selected_live_listing.get("square_feet") or 0
        )
        st.session_state.home_bedrooms_input = int(
            round(float(selected_live_listing.get("bedrooms") or 0))
        )
        st.session_state.home_bathrooms_input = float(
            selected_live_listing.get("bathrooms") or 0
        )
        st.session_state.home_age_input = 0

st.markdown("**Find an area or use your current location**")
st.caption(
    "Search by city, neighborhood, ZIP code, or address, or detect your device "
    "location. If access is denied, the selected neighborhood center remains active."
)

gps_data = browser_geolocation(
    key="ai_estate_browser_geolocation",
    default={"status": "idle"},
)
if isinstance(gps_data, dict) and gps_data.get("status") == "success":
    latitude = gps_data.get("latitude")
    longitude = gps_data.get("longitude")
    request_id = gps_data.get("request_id")
    current_coords = (latitude, longitude)
    if (
        latitude is not None
        and longitude is not None
        and request_id != st.session_state.get("_last_gps_request_id")
    ):
        if (
            -90 <= float(latitude) <= 90
            and -180 <= float(longitude) <= 180
        ):
            st.session_state["_last_gps_request_id"] = request_id
            st.session_state.last_gps_coords = current_coords
            st.session_state.selected_lat = float(latitude)
            st.session_state.selected_lon = float(longitude)
            st.session_state.location_query = "Current location"
            st.session_state.selected_location_label = "Current GPS location"
            st.session_state.selected_location_source = "Device GPS"

            try:
                geolocator = Nominatim(user_agent="ai_estate_property_valuation")
                address = geolocator.reverse(
                    (latitude, longitude), language="en", timeout=10
                )
                if address:
                    st.session_state.location_query = address.address
                    st.session_state.selected_location_label = address.address
                    st.session_state.selected_location_details = get_location_details(
                        address
                    )
            except Exception as error:
                st.warning(
                    "GPS coordinates were detected, but reverse address lookup "
                    f"failed: {error}"
                )
            st.success("Current device location detected and applied.")
    elif (
        latitude is not None
        and longitude is not None
        and not (-90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180)
    ):
        st.error("The browser returned coordinates outside the valid range.")
elif isinstance(gps_data, dict) and gps_data.get("status") == "error":
    st.warning(
        f"{gps_data.get('message', 'Location access failed')} "
        "The selected neighborhood or city center is still in use."
    )

location = st.text_input(
    "Area, neighborhood or ZIP",
    key="location_query",
    placeholder="City, neighborhood, ZIP code or address"
)

if st.button("Find area"):
    if location.strip():
        try:
            geolocator = Nominatim(user_agent="ai_estate_property_valuation")
            result = geolocator.geocode(location, timeout=10)
            if result:
                st.session_state.selected_lat = float(result.latitude)
                st.session_state.selected_lon = float(result.longitude)
                st.session_state.selected_location_label = result.address
                st.session_state.selected_location_source = "Address search result"
                st.session_state.selected_location_details = get_location_details(
                    result
                )
                st.session_state.last_gps_coords = None
                st.success("Area found and map updated.")
            else:
                st.warning("Location not found.")
        except Exception as e:
            st.error(f"Search failed: {e}")

if (
    st.session_state.selected_lat is not None
    and st.session_state.selected_lon is not None
):
    st.markdown("**Selected Location on Map**")
    location_label = str(
        st.session_state.get(
            "selected_location_label",
            st.session_state.location_query,
        )
    )
    location_source = str(
        st.session_state.get("selected_location_source", "Selected location")
    )
    location_details = st.session_state.get(
        "selected_location_details",
        get_location_details(None),
    )
    latitude = float(st.session_state.selected_lat)
    longitude = float(st.session_state.selected_lon)
    location_map = folium.Map(
        location=[
            latitude,
            longitude,
        ],
        zoom_start=17,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        "OpenStreetMap",
        name="Street map",
    ).add_to(location_map)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Satellite imagery",
    ).add_to(location_map)
    folium.Marker(
        location=[
            latitude,
            longitude,
        ],
        tooltip=location_label,
        popup=location_label,
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(location_map)
    folium.LayerControl(collapsed=True).add_to(location_map)
    st_folium(
        location_map,
        height=360,
        use_container_width=True,
        returned_objects=[],
    )
    maps_url = f"https://www.google.com/maps?q={latitude:.6f},{longitude:.6f}"
    st.markdown(
        f"**Area / neighborhood:** {escape(str(location_details.get('area') or 'Not available'))}  \n"
        f"**City / town:** {escape(str(location_details.get('city') or 'Not available'))}  \n"
        f"**Road / street:** {escape(str(location_details.get('road') or 'Not available'))}  \n"
        f"**State:** {escape(str(location_details.get('state') or 'Not available'))}  \n"
        f"**Postal code:** {escape(str(location_details.get('postcode') or 'Not available'))}  \n"
        f"**Property address / location:** {escape(location_label)}  \n"
        f"**Location source:** {escape(location_source)}  \n"
        f"**Coordinates:** {latitude:.6f}, {longitude:.6f}  \n"
        f"[Open this location in Google Maps]({maps_url})"
    )
    st.caption(
        "A locality-center pin is approximate, not an exact street address. "
        "Search for the full property address or use device GPS to locate the property more precisely."
    )

if asset_type == "Home":
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(
            "Living area (sq.ft)",
            min_value=0.0,
            step=50.0,
            key="home_area_input",
        )
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            step=1,
            key="home_bedrooms_input",
        )
    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            step=0.25,
            key="home_bathrooms_input",
        )
        age = st.number_input(
            "Property age (years)",
            min_value=0,
            step=1,
            key="home_age_input",
        )
    land_details = {}
else:
    st.info(
        "Land valuations use a separate land-trained model. Ensure its training "
        "dataset uses the same currency and local market as the target plot."
    )
    if not LAND_MODEL_PATH.is_file():
        st.warning(
            f"Dedicated land model not found at {LAND_MODEL_PATH}"
        )
        st.caption(
            "The automatic fallback trains from sample/demo data. Its estimates "
            "are for demonstration only, not real market valuations."
        )
        if st.button("⚡ Train Model & Auto-Fix Now"):
            try:
                with st.spinner("Training the dedicated land demo model..."):
                    train_land_model(
                        land_df,
                        model_path=LAND_MODEL_PATH,
                        target_currency="INR",
                        synthetic_data=True,
                    )
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Could not train the land demo model: {error}")
            else:
                st.session_state.allow_synthetic_land_model = True
                st.success(
                    "The demo model was trained. Its estimates are not real "
                    "market valuations."
                )
                st.rerun()

    land_col1, land_col2 = st.columns(2)
    with land_col1:
        area = st.number_input(
            "Plot area (sq.ft)",
            min_value=1.0,
            step=100.0,
            key="land_plot_area",
        )
        if not st.session_state.get("_land_frontage_default_applied"):
            if (
                st.session_state.get("_land_form_record_key") is None
                and st.session_state.get("land_frontage_width", 0.0) == 0.0
            ):
                st.session_state.land_frontage_width = 30.0
            st.session_state["_land_frontage_default_applied"] = True
        frontage = st.number_input(
            "Frontage width (ft)",
            min_value=0.0,
            step=5.0,
            key="land_frontage_width",
        )
        zoning = st.selectbox(
            "Zoning category",
            list(dict.fromkeys([
                "Residential", "Commercial", "Industrial", "Agricultural", "Mixed-Use",
                str(selected_property["zoning_category"]),
            ])),
            key="land_zoning",
        )
        topography = st.selectbox(
            "Topography",
            list(dict.fromkeys([
                "Flat", "Sloped", "Irregular",
                str(selected_property["topography_type"]),
            ])),
            key="land_topography",
        )
        soil_type = st.selectbox(
            "Soil type",
            list(dict.fromkeys([
                "Red", "Clay", "Sandy", "Black",
                str(selected_property["soil_type"]),
            ])),
            key="land_soil_type",
        )
    with land_col2:
        road_width = st.number_input(
            "Road width (ft)",
            min_value=0.0,
            step=5.0,
            key="land_road_width",
        )
        access_road_type = st.selectbox(
            "Access road type",
            list(dict.fromkeys([
                "Paved / National Highway", "Tar Road", "Dirt Road",
                str(selected_property["access_road_type"]),
            ])),
            key="land_access_road_type",
        )
        distance_to_main_road = st.number_input(
            "Distance to main road (m)",
            min_value=0.0,
            step=25.0,
            key="land_distance_to_main_road",
        )
        fsi_far_ratio = st.number_input(
            "FAR / FSI ratio",
            min_value=0.0,
            step=0.1,
            key="land_fsi_far_ratio",
        )
        corner_plot = st.checkbox("Corner plot", key="land_corner_plot")
        utilities_available = st.checkbox(
            "Water and power utilities available",
            key="land_utilities_available",
        )
    bedrooms = bathrooms = age = 0
    land_details = {
        "frontage_width_ft": frontage,
        "zoning_category": zoning,
        "topography_type": topography,
        "soil_type": soil_type,
        "corner_plot_flag": int(corner_plot),
        "fsi_far_ratio": fsi_far_ratio,
        "road_width_ft": road_width,
        "access_road_type": access_road_type,
        "distance_to_main_road_m": distance_to_main_road,
        "utilities_available": utilities_available,
    }

location_coordinates = (
    float(st.session_state.selected_lat),
    float(st.session_state.selected_lon),
)
micro_location_lookup = st.session_state.get("micro_location_lookup")
if st.button(
    "Calculate OSM micro-location score (1.5 km)",
    key="calculate_micro_location_score",
):
    update_micro_location_score(location_coordinates)

micro_location_lookup = st.session_state.get("micro_location_lookup")
has_current_micro_score = bool(
    micro_location_lookup
    and micro_location_lookup["coordinates"] == location_coordinates
)
if (
    st.session_state.get("_location_score_coordinates") != location_coordinates
    or "location_score_input" not in st.session_state
):
    default_location_score = (
        int(round(micro_location_lookup["score"]))
        if has_current_micro_score
        else min(
            10,
            max(
                2 if asset_type == "Land" else 1,
                int(round(float(selected_property["location_score"]))),
            ),
        )
    )
    st.session_state.location_score_input = default_location_score
    st.session_state["_location_score_coordinates"] = location_coordinates

location_score = st.slider(
    "Location score",
    min_value=1,
    max_value=10,
    key="location_score_input",
    help="OSM density-based score for schools, transit, hospitals, and commercial amenities, or adjust manually.",
)
location_score_uncalculated = (
    asset_type == "Land"
    and location_score == 1
    and not has_current_micro_score
)
if location_score_uncalculated:
    st.error(
        "Location score is set to 1 (uncalculated). This will heavily undervalue "
        "the plot in Vidyanagar, Hubballi."
    )
    if st.button(
        "Run OSM 1.5 km score calculation first",
        key="run_osm_score_before_land_valuation",
    ):
        update_micro_location_score(location_coordinates)

if has_current_micro_score:
    st.caption(
        "OpenStreetMap density estimate within 1.5 km "
        f"(score {micro_location_lookup['score']:.1f}/10). Counts: "
        + ", ".join(
            f"{category.lower()} {details['count']}"
            for category, details in micro_location_lookup["results"].items()
        )
        + ". This is a heuristic based on mapped amenities, not a safety or quality rating."
    )

if asset_type == "Land":
    land_model_features = {
        "plot_area_sqft": area,
        "frontage_width_ft": frontage,
        "topography_type": topography,
        "soil_type": soil_type,
        "corner_plot_flag": int(corner_plot),
        "zoning_category": zoning,
        "fsi_far_ratio": fsi_far_ratio,
        "road_width_ft": road_width,
        "access_road_type": access_road_type,
        "distance_to_main_road_m": distance_to_main_road,
        "water_power_utility_access": int(utilities_available),
        "lat": float(st.session_state.selected_lat),
        "long": float(st.session_state.selected_lon),
        "micro_location_score": float(location_score),
    }

if asset_type == "Home":
    st.caption("Parking information is not present in this dataset.")

st.write("")

with st.container(border=True):
    st.markdown(
        '<div class="section-title">Property Gallery</div>',
        unsafe_allow_html=True
    )

    images = get_property_images(dataset_identifier)
    photo_category = st.selectbox(
        "Category for captured photos",
        options=["Exterior", "Kitchen", "Living room", "Bedroom", "Bathroom", "Land / site", "Other"],
    )
    if images:
        st.markdown(
            f'<div class="section-description">{len(images)} property image(s) available</div>',
            unsafe_allow_html=True
        )
        image_columns = st.columns(min(len(images), 3))
        for index, image_path in enumerate(images):
            with image_columns[index % 3]:
                st.image(
                    image_path,
                    caption=os.path.basename(image_path),
                    use_container_width=True
                )
                with open(image_path, "rb") as image_file:
                    render_photo_quality_badge(image_file.read())
    else:
        st.caption("No gallery images are available for this dataset record.")

    if image_only_property_ids:
        with st.expander("Images for records without valuation data", expanded=False):
            image_only_id = st.selectbox(
                "Property ID",
                options=image_only_property_ids,
                key="image_only_property_id",
            )
            st.caption(
                f"Property ID {image_only_id} has linked photos, but its property "
                "details are blank, so it is not available for valuation yet."
            )
            image_only_images = get_property_images(image_only_id)
            image_only_columns = st.columns(min(len(image_only_images), 3))
            for index, image_path in enumerate(image_only_images):
                with image_only_columns[index % len(image_only_columns)]:
                    st.image(
                        image_path,
                        caption=os.path.basename(image_path),
                        use_container_width=True,
                    )
                    with open(image_path, "rb") as image_file:
                        render_photo_quality_badge(image_file.read())

    st.info(
        "For Android camera capture, use the HTTPS ngrok URL and allow camera access in your browser. "
        "This improves on-site inspection quality and field valuation accuracy."
    )

    uploaded_images = st.file_uploader(
        "Add property photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="valuation_images",
    )
    camera_photo = st.camera_input(
        "Capture a live property or land photo",
        key="camera_photo",
    )
    if camera_photo:
        st.caption("Camera snapshot ready for analysis with the next valuation.")

    selected_land_sample_photos = []
    if asset_type == "Land":
        sample_land_photo_paths = {
            label: os.path.join(images_folder, "land_samples", filename)
            for label, filename in [
                ("Bare soil plot (synthetic demo)", "sample_bare_plot.jpg"),
                ("Grassy plot (synthetic demo)", "sample_grassy_plot.jpg"),
                ("Roadside plot (synthetic demo)", "sample_roadside_plot.jpg"),
            ]
        }
        sample_land_photo_paths = {
            label: path
            for label, path in sample_land_photo_paths.items()
            if os.path.isfile(path)
        }
        if sample_land_photo_paths:
            selected_land_sample_photos = st.multiselect(
                "Optional sample land photos for demo analysis",
                options=list(sample_land_photo_paths),
                help=(
                    "These generated illustrations demonstrate land-photo cues. "
                    "They are not photos of your selected property."
                ),
                key="land_sample_photos",
            )
            if selected_land_sample_photos:
                st.info(
                    "Synthetic demo images will be analyzed alongside any photos "
                    "you uploaded. They provide visual cues only and do not change "
                    "the land model's price estimate."
                )
                sample_columns = st.columns(
                    min(len(selected_land_sample_photos), 3)
                )
                for index, label in enumerate(selected_land_sample_photos):
                    with sample_columns[index % len(sample_columns)]:
                        st.image(
                            sample_land_photo_paths[label],
                            caption=label,
                            use_container_width=True,
                        )
                        with open(sample_land_photo_paths[label], "rb") as sample_file:
                            render_photo_quality_badge(sample_file.read())

    inspection_photos = [
        {
            "filename": image.name,
            "category": photo_category,
            "content": image.getvalue(),
        }
        for image in uploaded_images
    ]
    if camera_photo:
        inspection_photos.append({
            "filename": "camera-capture.jpg",
            "category": photo_category,
            "content": camera_photo.getvalue(),
        })
    if not inspection_photos:
        for image_path in images:
            with open(image_path, "rb") as image_file:
                inspection_photos.append({
                    "filename": os.path.basename(image_path),
                    "category": "Existing property gallery",
                    "content": image_file.read(),
                })

    if uploaded_images or camera_photo:
        st.markdown("**Photo quality preview**")
        photo_columns = st.columns(min(len(inspection_photos), 3))
        for index, photo in enumerate(inspection_photos):
            if photo.get("category") == "Existing property gallery":
                continue
            with photo_columns[index % len(photo_columns)]:
                st.image(
                    photo["content"],
                    caption=photo["filename"],
                    use_container_width=True,
                )
                render_photo_quality_badge(photo["content"])

    render_room_measurement(inspection_photos, dataset_identifier)

    if st.button("Save inspection locally"):
        try:
            saved_inspection = save_inspection(
                {
                    "asset_type": asset_type,
                    "property_record": property_label,
                    "location": st.session_state.location_query,
                    "latitude": float(st.session_state.selected_lat),
                    "longitude": float(st.session_state.selected_lon),
                    "area_sqft": float(area),
                    "area_kind": "living area" if asset_type == "Home" else "plot area",
                    "bedrooms": int(bedrooms) if asset_type == "Home" else None,
                    "bathrooms": float(bathrooms) if asset_type == "Home" else None,
                    "property_age_years": int(age) if asset_type == "Home" else None,
                    "location_score": int(location_score),
                    **land_details,
                },
                inspection_photos,
            )
            st.success(
                f"Inspection saved locally with {len(saved_inspection['photos'])} photo(s)."
            )
        except Exception as e:
            st.error(f"Could not save inspection: {e}")

st.write("")

with st.container(border=True):
    action_title = "Land Valuation & Inspection" if asset_type == "Land" else "AI Property Valuation"
    action_button = "Estimate Land Value  →" if asset_type == "Land" else "Estimate Property Value  →"
    action_description = (
        "Estimate land value per square foot and total parcel value using the dedicated land model."
        if asset_type == "Land"
        else "Analyze the selected photos and property details."
    )
    st.markdown(f'<div class="section-title">{action_title}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-description">{action_description}</div>',
        unsafe_allow_html=True
    )

    if st.button(
        action_button,
        type="primary",
        disabled=location_score_uncalculated,
    ):
        photo_sources = [photo["content"] for photo in inspection_photos]
        if asset_type == "Land":
            for label in selected_land_sample_photos:
                sample_path = sample_land_photo_paths[label]
                with open(sample_path, "rb") as sample_file:
                    photo_sources.append(sample_file.read())
        visual_analysis = None
        if photo_sources:
            try:
                if asset_type == "Land":
                    from vision_module import analyze_land_images

                    visual_analysis = analyze_land_images(photo_sources)
                else:
                    from vision_module import analyze_images

                    visual_analysis = analyze_images(photo_sources)
            except Exception as e:
                st.warning(
                    f"Photo analysis is unavailable; continuing with the entered "
                    f"property details. {e}"
                )

        image_impact = None
        if (
            asset_type == "Home"
            and visual_analysis
            and visual_analysis["room_type"] == "Open land / landscape"
        ):
            st.warning(
                "The selected photo looks like open land. Home valuation was "
                "skipped; switch Asset type to Land to enter parcel details and "
                "use the dedicated land model."
            )
        elif asset_type == "Land":
            try:
                if not LAND_MODEL_PATH.is_file():
                    with st.spinner(
                        "Training a demo land model from the loaded sample records..."
                    ):
                        train_land_model(
                            land_df,
                            model_path=LAND_MODEL_PATH,
                            target_currency="INR",
                            synthetic_data=True,
                        )
                    st.session_state.allow_synthetic_land_model = True
                land_bundle = load_land_model(
                    LAND_MODEL_PATH,
                    allow_synthetic=st.session_state.get(
                        "allow_synthetic_land_model",
                        False,
                    ),
                )
                land_property_data = pd.DataFrame(
                    [land_model_features],
                    columns=LAND_FEATURES,
                )
                price_per_sqft = float(land_bundle["model"].predict(land_property_data)[0])
                if not math.isfinite(price_per_sqft) or price_per_sqft <= 0:
                    raise ValueError("The land model returned an invalid price per square foot.")
            except FileNotFoundError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Land valuation failed: {error}")
            else:
                if land_bundle.get("synthetic_data", False):
                    st.warning(
                        "This estimate uses a synthetic demo model and must not "
                        "be treated as a real property valuation."
                    )
                contributions = []
                try:
                    contributions = explain_land_prediction(
                        land_bundle,
                        land_property_data,
                    )
                except (ImportError, RuntimeError, TypeError, ValueError) as error:
                    st.warning(f"Land TreeSHAP explanations are unavailable: {error}")
                target_currency = land_bundle["target_currency"]
                display_rate = (
                    1.0 if target_currency == "INR" else usd_inr_rate
                )
                total_price = price_per_sqft * area
                interval_error = land_bundle.get("interval_error")
                total_interval_error = (
                    interval_error * area
                    if interval_error is not None
                    else None
                )
                st.session_state.last_land_valuation = {
                    "property_name": property_label,
                    "plot_area_sqft": float(area),
                    "price_per_sqft": price_per_sqft,
                    "total_price": total_price,
                    "target_currency": target_currency,
                    "interval_error": total_interval_error,
                    "feature_contributions": contributions,
                    "model_metrics": land_bundle.get("metrics", {}),
                }
                if display_rate is None:
                    st.error(
                        "The land model predicts USD, but INR conversion is "
                        "unavailable. The model-currency estimate is shown instead."
                    )
                    display_rate = 1.0
                    currency_label = "USD"
                else:
                    currency_label = "INR"
                st.metric(
                    f"Total Land Parcel Value ({currency_label})",
                    format_indian_rupees(total_price * display_rate)
                    if currency_label == "INR"
                    else f"${total_price:,.2f}",
                )
                per_sqft_value = price_per_sqft * display_rate
                st.metric(
                    f"Estimated Price per sq. ft. ({currency_label})",
                    format_indian_rupees(per_sqft_value)
                    if currency_label == "INR"
                    else f"${price_per_sqft:,.2f}",
                )
                st.caption(
                    "Land value is plot area multiplied by the model's estimated "
                    f"price per square foot. Training target currency: {target_currency}."
                )
                if interval_error is not None:
                    st.caption(
                        "90% Residual Value Range (held-out estimate): "
                        + (
                            f"{format_indian_rupees(max(0, total_price - total_interval_error) * display_rate)}"
                            " to "
                            f"{format_indian_rupees((total_price + total_interval_error) * display_rate)}"
                            if currency_label == "INR"
                            else f"${max(0, total_price - total_interval_error):,.2f} to "
                            f"${total_price + total_interval_error:,.2f}"
                        )
                    )
                if contributions and (
                    target_currency == "INR" or usd_inr_rate is not None
                ):
                    st.markdown("**TreeSHAP land value drivers**")
                    st.caption(
                        "Contributions are estimated effects on total plot value "
                        "relative to the model baseline."
                    )
                    render_land_contributions(
                        contributions,
                        area,
                        target_currency,
                        usd_inr_rate,
                    )
                elif contributions:
                    st.info(
                        "TreeSHAP contributions are available after the USD-to-INR "
                        "exchange rate is restored."
                    )
                training_metrics = land_bundle.get("metrics", {})
                if training_metrics:
                    st.caption(
                        "Land model holdout metrics — "
                        f"R² {training_metrics.get('r2', float('nan')):.3f}, "
                        f"MAE {training_metrics.get('mae', float('nan')):,.2f} "
                        f"{target_currency}/sq.ft."
                    )
        else:
            property_data = pd.DataFrame(
                [{column: selected_property.get(column, 0.0) for column in model_features}],
                columns=model_features,
            )
            property_data = property_data.apply(
                pd.to_numeric,
                errors="coerce",
            ).replace(
                [float("inf"), -float("inf")],
                pd.NA,
            ).fillna(0.0)
            property_data.at[0, "sqft_living"] = area
            property_data.at[0, "bedrooms"] = bedrooms
            property_data.at[0, "bathrooms"] = bathrooms
            property_data.at[0, "house_age"] = age
            property_data.at[0, "lat"] = st.session_state.selected_lat
            property_data.at[0, "long"] = st.session_state.selected_lon
            property_data.at[0, "location_score"] = location_score
            if visual_analysis:
                for column in VISUAL_COLUMNS:
                    if column in property_data:
                        property_data.at[0, column] = visual_analysis["features"].get(
                            column,
                            0.0,
                        )

            try:
                predicted_price = float(model.predict(property_data)[0])
                if not math.isfinite(predicted_price):
                    raise ValueError("The model returned an invalid estimate.")
                if visual_analysis:
                    no_photo_data = property_data.copy()
                    for column in VISUAL_COLUMNS:
                        if column in no_photo_data:
                            no_photo_data.at[0, column] = 0.0
                    image_impact = predicted_price - float(
                        model.predict(no_photo_data)[0]
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            else:
                lower_bound = max(0.0, predicted_price - model_bundle["interval_error"])
                upper_bound = predicted_price + model_bundle["interval_error"]
                feature_contributions = []
                try:
                    feature_contributions = explain_xgboost_prediction(
                        model,
                        property_data,
                    )
                except (TypeError, ValueError, RuntimeError) as e:
                    st.warning(f"Per-property feature explanations are unavailable: {e}")
                valuation_attributes = {
                    "Property type": asset_type,
                    "Property ID": (
                        normalize_property_id(source_identifier)
                        if pd.notna(source_identifier)
                        else "Not available"
                    ),
                    "Living area (sq.ft)": f"{area:,.0f}",
                    "Bedrooms": str(bedrooms),
                    "Bathrooms": f"{bathrooms:g}",
                    "Age (years)": str(age),
                    "Locality": str(selected_property.get("area_name", "Not available")),
                    "Location score": f"{location_score}/10",
                    "Latitude": f"{st.session_state.selected_lat:.6f}",
                    "Longitude": f"{st.session_state.selected_lon:.6f}",
                    "Inspection timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
                if visual_analysis:
                    valuation_attributes["Property finish & lighting score"] = (
                        f"{visual_analysis['features']['image_finish_score']:.1f}/5"
                    )
                    valuation_attributes["Photo clarity score"] = (
                        f"{visual_analysis['quality_score']:.0f}/100"
                    )
                st.session_state.last_valuation = {
                    "property_name": property_label,
                    "predicted_value": predicted_price,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "area_name": str(selected_property.get("area_name", "")),
                    "latitude": float(st.session_state.selected_lat),
                    "longitude": float(st.session_state.selected_lon),
                    "attributes": valuation_attributes,
                    "feature_contributions": feature_contributions,
                    "photos": inspection_photos,
                    "model_features": {
                        column: float(value)
                        for column, value in property_data.iloc[0].items()
                    },
                    "interval_error": float(model_bundle["interval_error"]),
                    "visual_quality_score": (
                        float(visual_analysis["quality_score"])
                        if visual_analysis
                        else None
                    ),
                    "visual_photo_analyses": (
                        [
                            {
                                "room_type": photo_result.get("room_type"),
                                "quality_score": photo_result.get("quality_score"),
                            }
                            for photo_result in visual_analysis.get("photos", [])
                        ][:len(inspection_photos)]
                        if visual_analysis
                        else []
                    ),
                }
                st.session_state.pop("generated_ai_listing", None)
                if usd_inr_rate is not None:
                    render_valuation_gauge(
                        st.session_state.last_valuation,
                        usd_inr_rate,
                        key="valuation_gauge_primary",
                    )
                    st.caption(
                        "Model output converted at 1 USD = "
                        f"₹{usd_inr_rate:,.2f} ({exchange_rate_updated_at})."
                    )
                else:
                    st.error(
                        "The prediction was calculated, but its INR value cannot be "
                        "shown until the exchange-rate service is available."
                    )
                if feature_contributions and usd_inr_rate is not None:
                    st.markdown("**TreeSHAP value drivers (this estimate)**")
                    st.caption(
                        "Positive contributions raise this estimate; negative "
                        "contributions lower it."
                    )
                    render_feature_contributions(
                        feature_contributions,
                        usd_inr_rate,
                        key="valuation_feature_contributions",
                    )
        if visual_analysis and asset_type == "Land":
            land_cues = visual_analysis.get("land_visual_cues", {})
            st.markdown("**Land photo analysis**")
            cue_columns = st.columns(3)
            cue_columns[0].metric(
                "Green vegetation pixels",
                f"{land_cues.get('vegetation_area_pct', 0.0):.1f}%",
            )
            cue_columns[1].metric(
                "Earth-tone pixels",
                f"{land_cues.get('earth_tone_area_pct', 0.0):.1f}%",
            )
            cue_columns[2].metric(
                "Image edge density",
                f"{land_cues.get('edge_density_pct', 0.0):.1f}%",
            )
            st.caption(
                "Image color and edge-density heuristics only; they do not detect "
                "legal plot boundaries, soil quality, or land area. Recognized labels: "
                + ", ".join(land_cues.get("recognized_tags", []))
            )
        elif visual_analysis:
            st.markdown("**Image insights**")
            st.write(
                f"Likely view: {visual_analysis['room_type']} · "
                f"Photo condition score: {visual_analysis['quality_score']:.0f}/100 · "
                "Property finish & lighting score: "
                f"{visual_analysis['features']['image_finish_score']:.1f}/5"
            )
            st.caption(
                "Recognized visual features: "
                + ", ".join(visual_analysis["key_visual_features"])
            )
            if image_impact is not None:
                direction = "above" if image_impact >= 0 else "below"
                if usd_inr_rate is not None:
                    st.caption(
                        f"Model-estimated photo impact: {direction} the no-photo "
                        "baseline by "
                        f"{format_indian_rupees(abs(image_impact) * usd_inr_rate)}."
                    )
            st.caption("Photo condition is a lighting and sharpness heuristic, not a property inspection.")
# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    st.write("")

    with st.expander("Saved inspections", expanded=False):
        saved_records = list_inspections()
        if not saved_records:
            st.caption("No inspection sessions have been saved yet.")
        else:
            record_options = {
                f"{record['created_at']} · {record['asset_type']} · {record['location']} · {record['inspection_id'][:8]}": record
                for record in saved_records
            }
            selected_record_label = st.selectbox(
                "Inspection session",
                options=list(record_options),
                key="saved_inspection_record",
            )
            selected_record = record_options[selected_record_label]
            st.caption(f"{len(selected_record['photos'])} saved photo(s)")
            if selected_record["photos"]:
                saved_photo_columns = st.columns(min(3, len(selected_record["photos"])))
                session_dir = DEFAULT_INSPECTIONS_DIR / selected_record["inspection_id"]
                for photo_index, photo in enumerate(selected_record["photos"]):
                    with saved_photo_columns[photo_index % len(saved_photo_columns)]:
                        st.image(
                            str(session_dir / photo["file"]),
                            caption=photo["category"],
                            use_container_width=True,
                        )
            st.download_button(
                "Download inspection metadata",
                data=json.dumps(selected_record, indent=2, ensure_ascii=False),
                file_name=f"inspection-{selected_record['inspection_id']}.json",
                mime="application/json",
            )


if (
    st.session_state.selected_lat is not None
    and st.session_state.selected_lon is not None
):
    st.write("")


    # ABOUT PROJECT

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">About the Project</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-description">

            This application demonstrates an AI-based
            real-estate valuation system.

            The valuation combines numeric property features with ResNet photo
            features when photos are available. An optional OpenStreetMap
            micro-location score summarizes amenities within 1.5 km:

            <br>

            • sqft_living, bedrooms, and bathrooms<br>
            • house_age and location coordinates (lat, long)<br>
            • Other numeric dataset features and the editable location score

            <br>

            Parking is not present in the dataset. Nearby sales comparables come
            from local historical records and are not live listings.

            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")
nearby_tab, report_tab, photo_preview_tab, ai_studio_tab, market_tab, investment_tab, inspection_tab = st.tabs(
    [
        "🏡 Nearby Properties & Land for Sale",
        "Valuation",
        "Photo Preview",
        "AI Studio",
        "Market Trends",
        "Investment Calculator",
        "Inspection Logs",
    ]
)

with nearby_tab:
    st.subheader("🏡 Nearby Properties & Land for Sale")
    dataset_tab, live_tab, combined_tab = st.tabs(
        [
            "🏠 Dataset Listings (Internal)",
            "🌐 Live Market Listings",
            "🔀 Combined View",
        ]
    )

    nearby_latitude = float(st.session_state.selected_lat)
    nearby_longitude = float(st.session_state.selected_lon)
    nearby_location_name = str(
        st.session_state.get("selected_location_label")
        or st.session_state.get("location_query")
        or "selected location"
    )
    location_details_for_live_api = st.session_state.get(
        "selected_location_details", get_location_details(None)
    )
    location_city = str(
        location_details_for_live_api.get("city")
        or location_details_for_live_api.get("area")
        or "Unknown"
    )

    radius_km = st.select_slider(
        "Search radius",
        options=[1.5, 3.0, 5.0],
        value=1.5,
        format_func=lambda value: f"{value:.1f} km",
        key="nearby_property_radius_km",
    )

with dataset_tab:
    st.caption(
        "Local property records and sample land dataset. "
        "These are not verified active listings or live market offers."
    )
    try:
        all_nearby_records = nearby_property_records(
            data,
            land_df,
            nearby_latitude,
            nearby_longitude,
            usd_inr_rate,
        )
    except (TypeError, ValueError, OverflowError) as error:
        st.error(f"Could not search nearby local records: {error}")
    else:
        within_radius = all_nearby_records[
            all_nearby_records["distance_km"] <= radius_km
        ].copy()
        used_nearest_fallback = within_radius.empty
        if used_nearest_fallback:
            nearby_records = all_nearby_records.head(3).copy()
            st.info(
                f"No matching property listings found within {radius_km:.1f} km "
                f"of {nearby_location_name}. Showing the nearest 3 available "
                "comparables instead."
            )
        else:
            nearby_records = within_radius
            st.write(
                f"Found **{len(nearby_records)}** local dataset record(s) within "
                f"**{radius_km:.1f} km** of **{nearby_location_name}**."
            )

        if nearby_records.empty:
            st.info("No property or land records with valid coordinates are available.")
        else:
            if usd_inr_rate is None and (
                nearby_records["asset_type"] == "Home"
            ).any():
                st.warning(
                    "Home prices are recorded in USD; their INR amounts are "
                    "unavailable until the exchange-rate service responds. Land "
                    "sample prices are shown in INR."
                )

            table_records = nearby_records.copy()
            table_records["total_price_inr"] = table_records[
                "price_total_inr"
            ].map(
                lambda value: (
                    format_indian_rupees(value)
                    if value is not None and pd.notna(value)
                    else "Unavailable"
                )
            )
            table_records["price_per_sqft_inr"] = table_records[
                "price_per_sqft_inr"
            ].map(
                lambda value: (
                    format_indian_rupees(value)
                    if value is not None and pd.notna(value)
                    else "Unavailable"
                )
            )
            table_records["frontage_width_ft"] = table_records[
                "frontage_width_ft"
            ].map(lambda value: f"{value:,.1f} ft" if pd.notna(value) else "N/A")
            st.dataframe(
                table_records[
                    [
                        "asset_type",
                        "property_id",
                        "area_name",
                        "distance_km",
                        "plot_area_sqft",
                        "frontage_width_ft",
                        "total_price_inr",
                        "price_per_sqft_inr",
                    ]
                ].rename(
                    columns={
                        "asset_type": "Property type",
                        "property_id": "Record ID",
                        "area_name": "Locality / address",
                        "distance_km": "Distance (km)",
                        "plot_area_sqft": "Plot area (sq.ft)",
                        "frontage_width_ft": "Frontage",
                        "total_price_inr": "Total price (INR)",
                        "price_per_sqft_inr": "Price per sq.ft (INR)",
                    }
                ).style.format(
                    {
                        "Distance (km)": "{:.2f}",
                        "Plot area (sq.ft)": "{:,.0f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            nearby_map = make_nearby_listings_map(
                nearby_records,
                nearby_latitude,
                nearby_longitude,
                nearby_location_name,
            )
            nearby_map_state = st_folium(
                nearby_map,
                height=470,
                use_container_width=True,
                returned_objects=["last_object_clicked"],
                key="nearby_property_listings_map",
            )

            st.markdown("#### Side-by-side comparable matrix")
            compatible_records = nearby_records[
                nearby_records["asset_type"] == asset_type
            ]
            comparison_options = {
                (
                    f"{row['record']} · {row['distance_km']:.2f} km · "
                    f"{row['property_id']}"
                ): row_index
                for row_index, row in compatible_records.iterrows()
            }
            clicked_comparison = None
            clicked_marker = (
                nearby_map_state.get("last_object_clicked")
                if isinstance(nearby_map_state, dict)
                else None
            )
            if isinstance(clicked_marker, dict):
                clicked_lat = clicked_marker.get("lat")
                clicked_lon = clicked_marker.get("lng")
                if clicked_lat is not None and clicked_lon is not None:
                    for label, row_index in comparison_options.items():
                        candidate = nearby_records.loc[row_index]
                        distance_m = geodesic(
                            (float(clicked_lat), float(clicked_lon)),
                            (
                                float(candidate["latitude"]),
                                float(candidate["longitude"]),
                            ),
                        ).meters
                        if distance_m <= 50:
                            clicked_comparison = label
                            break
            selected_comparison_labels = st.multiselect(
                "Select records shown on the map to compare",
                options=list(comparison_options),
                default=[clicked_comparison] if clicked_comparison else [],
                key="nearby_comparison_selection",
            )
            if selected_comparison_labels:
                current_total_inr = None
                current_area_sqft = float(area)
                current_value = st.session_state.get("last_valuation")
                if (
                    asset_type == "Home"
                    and current_value
                    and usd_inr_rate is not None
                ):
                    current_total_inr = (
                        float(current_value["predicted_value"]) * usd_inr_rate
                    )
                    current_area_sqft = float(
                        str(
                            current_value["attributes"].get(
                                "Living area (sq.ft)",
                                area,
                            )
                        ).replace(",", "")
                    )
                elif asset_type == "Land":
                    current_land_value = st.session_state.get("last_land_valuation")
                    if current_land_value:
                        if current_land_value["target_currency"] == "INR":
                            current_total_inr = float(current_land_value["total_price"])
                        elif usd_inr_rate is not None:
                            current_total_inr = (
                                float(current_land_value["total_price"])
                                * usd_inr_rate
                            )
                        current_area_sqft = float(
                            current_land_value.get("plot_area_sqft", area)
                        )
                current_ppsf_inr = (
                    current_total_inr / current_area_sqft
                    if current_total_inr is not None and current_area_sqft > 0
                    else None
                )
                matrix_rows = [
                    {
                        "Property": "Current property",
                        "Distance (km)": 0.0,
                        "Area (sq.ft)": current_area_sqft,
                        "Total price (INR)": current_total_inr,
                        "Price / sq.ft (INR)": current_ppsf_inr,
                        "Price variance (INR)": 0.0 if current_total_inr is not None else None,
                        "Area difference (sq.ft)": 0.0,
                        "Price / sq.ft difference (INR)": 0.0,
                    }
                ]
                for label in selected_comparison_labels:
                    comparable = nearby_records.loc[
                        comparison_options[label]
                    ]
                    comparable_area = float(comparable["comparison_area_sqft"])
                    comparable_total = comparable["price_total_inr"]
                    comparable_ppsf = comparable["price_per_sqft_inr"]
                    matrix_rows.append(
                        {
                            "Property": str(comparable["property_id"]),
                            "Distance (km)": float(comparable["distance_km"]),
                            "Area (sq.ft)": comparable_area,
                            "Total price (INR)": comparable_total,
                            "Price / sq.ft (INR)": comparable_ppsf,
                            "Price variance (INR)": (
                                float(comparable_total) - current_total_inr
                                if comparable_total is not None
                                and current_total_inr is not None
                                else None
                            ),
                            "Area difference (sq.ft)": (
                                comparable_area - current_area_sqft
                            ),
                            "Price / sq.ft difference (INR)": (
                                float(comparable_ppsf) - current_ppsf_inr
                                if comparable_ppsf is not None
                                and current_ppsf_inr is not None
                                else None
                            ),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(matrix_rows).style.format(
                        {
                            "Distance (km)": "{:.2f}",
                            "Area (sq.ft)": "{:,.0f}",
                            "Total price (INR)": lambda value: (
                                format_indian_rupees(value) if pd.notna(value) else "N/A"
                            ),
                            "Price / sq.ft (INR)": lambda value: (
                                format_indian_rupees(value) if pd.notna(value) else "N/A"
                            ),
                            "Price variance (INR)": lambda value: (
                                format_indian_rupees(value) if pd.notna(value) else "N/A"
                            ),
                            "Area difference (sq.ft)": "{:+,.0f}",
                            "Price / sq.ft difference (INR)": lambda value: (
                                format_indian_rupees(value) if pd.notna(value) else "N/A"
                            ),
                        },
                        na_rep="N/A",
                    ).background_gradient(
                        cmap="RdYlGn_r",
                        subset=[
                            "Price variance (INR)",
                            "Price / sq.ft difference (INR)",
                        ],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if current_total_inr is None:
                    st.caption(
                        "Run a valuation to compare price variance against the current "
                        "property. Nearby home prices require the live INR conversion rate."
                    )

            listing_options = [
                (
                    f"{row['record']} · {row['distance_km']:.2f} km · "
                    f"{row['property_id']}"
                )
                for row in nearby_records.to_dict(orient="records")
            ]
            selected_listing_option = st.selectbox(
                "Choose a nearby record to inspect",
                listing_options,
                key="nearby_listing_to_inspect",
            )
            selected_listing_index = listing_options.index(
                selected_listing_option
            )
            selected_listing = nearby_records.iloc[selected_listing_index]
            st.button(
                "Inspect / Load into valuation form",
                on_click=select_nearby_listing,
                args=(
                    selected_listing["asset_type"],
                    int(selected_listing["row_key"]),
                ),
                key="load_nearby_listing",
            )

with report_tab:
    st.subheader("Valuation report and locality amenities")
    last_valuation = st.session_state.get("last_valuation")
    if last_valuation:
        if usd_inr_rate is None:
            st.error("INR conversion is unavailable; the exchange-rate service failed.")
        else:
            render_valuation_gauge(
                last_valuation,
                usd_inr_rate,
                key="valuation_gauge_report",
            )
        finish_score = last_valuation["attributes"].get(
            "Property finish & lighting score"
        )
        if finish_score:
            st.metric("Property finish & lighting score", finish_score)
        if last_valuation["feature_contributions"] and usd_inr_rate is not None:
            st.markdown("**TreeSHAP value drivers**")
            st.caption(
                "Positive contributions raise this estimate; negative "
                "contributions lower it."
            )
            render_feature_contributions(
                last_valuation["feature_contributions"],
                usd_inr_rate,
                key="report_feature_contributions",
            )
        if usd_inr_rate is not None:
            try:
                summary_pdf = generate_valuation_summary_pdf(
                    property_name=last_valuation["property_name"],
                    predicted_value=last_valuation["predicted_value"],
                    lower_bound=last_valuation["lower_bound"],
                    upper_bound=last_valuation["upper_bound"],
                    attributes=last_valuation["attributes"],
                    latitude=last_valuation["latitude"],
                    longitude=last_valuation["longitude"],
                    exchange_rate=usd_inr_rate,
                    inspection_timestamp=last_valuation["attributes"][
                        "Inspection timestamp"
                    ],
                    photo_quality_summary=last_valuation["attributes"].get(
                        "Photo clarity score",
                        "No photo analysis",
                    ),
                    location_score=last_valuation["attributes"].get(
                        "Location score",
                        "Not available",
                    ),
                )
            except Exception as error:
                st.error(f"Could not generate the PDF valuation summary: {error}")
            else:
                st.download_button(
                    "Download Valuation Report",
                    data=summary_pdf,
                    file_name="ai-estate-valuation-summary.pdf",
                    mime="application/pdf",
                    key="valuation_summary_download",
                )

            try:
                valuation_pdf = generate_valuation_pdf(
                    property_name=last_valuation["property_name"],
                    predicted_value=last_valuation["predicted_value"],
                    lower_bound=last_valuation["lower_bound"],
                    upper_bound=last_valuation["upper_bound"],
                    attributes=last_valuation["attributes"],
                    area_name=last_valuation["area_name"],
                    latitude=last_valuation["latitude"],
                    longitude=last_valuation["longitude"],
                    feature_contributions=last_valuation["feature_contributions"],
                    photos=last_valuation["photos"],
                    exchange_rate=usd_inr_rate,
                    exchange_rate_updated_at=exchange_rate_updated_at,
                )
            except Exception as error:
                st.error(f"Could not generate the valuation PDF: {error}")
            else:
                st.download_button(
                    "Download detailed valuation report (PDF)",
                    data=valuation_pdf,
                    file_name="ai-estate-valuation-report.pdf",
                    mime="application/pdf",
                    key="valuation_report_download",
                )
    else:
        st.info("Run a property valuation to generate its report and explanations.")

    last_land_valuation = st.session_state.get("last_land_valuation")
    if last_land_valuation:
        st.markdown("**Latest dedicated land estimate**")
        target_currency = last_land_valuation["target_currency"]
        conversion_rate = (
            1.0 if target_currency == "INR" else usd_inr_rate
        )
        if conversion_rate is None:
            st.metric(
                "Estimated total land value (USD)",
                f"${last_land_valuation['total_price']:,.2f}",
            )
            st.metric(
                "Estimated land value per sq.ft (USD)",
                f"${last_land_valuation['price_per_sqft']:,.2f}",
            )
            st.warning(
                "The USD-to-INR service is unavailable; showing the model's "
                "original USD currency."
            )
        else:
            st.metric(
                "Estimated total land value (INR)",
                format_indian_rupees(
                    last_land_valuation["total_price"] * conversion_rate
                ),
            )
            st.metric(
                "Estimated land value per sq.ft (INR)",
                format_indian_rupees(
                    last_land_valuation["price_per_sqft"] * conversion_rate
                ),
            )
        st.caption(
            f"{last_land_valuation['property_name']} · "
            f"{last_land_valuation['plot_area_sqft']:,.0f} sq.ft · "
            f"training target {target_currency}/sq.ft"
        )
        land_metrics = last_land_valuation.get("model_metrics", {})
        if land_metrics:
            st.caption(
                f"Holdout R² {land_metrics['r2']:.3f} · "
                f"MAE {land_metrics['mae']:,.2f} {target_currency}/sq.ft · "
                f"RMSE {land_metrics['rmse']:,.2f} {target_currency}/sq.ft"
            )
        if last_land_valuation["feature_contributions"]:
            if target_currency == "INR" or usd_inr_rate is not None:
                st.markdown("**TreeSHAP land value drivers**")
                render_land_contributions(
                    last_land_valuation["feature_contributions"],
                    last_land_valuation["plot_area_sqft"],
                    target_currency,
                    usd_inr_rate,
                )
            else:
                st.info(
                    "Land TreeSHAP contributions in INR are unavailable until "
                    "the exchange-rate service responds."
                )

with photo_preview_tab:
    st.subheader("Interactive property photo preview")
    latest_photo_valuation = st.session_state.get("last_valuation")
    if latest_photo_valuation and latest_photo_valuation.get("photos"):
        preview_photos = latest_photo_valuation["photos"]
        preview_analyses = latest_photo_valuation.get(
            "visual_photo_analyses",
            [],
        )
        st.caption(
            f"Latest valuation: {latest_photo_valuation['property_name']} · "
            f"{len(preview_photos)} image(s)"
        )
    else:
        preview_photos = inspection_photos
        preview_analyses = []

    if not preview_photos:
        st.info("Add or select property photos to open the comparison preview.")
    else:
        gallery_columns = st.columns(min(3, len(preview_photos)))
        for index, photo in enumerate(preview_photos):
            with gallery_columns[index % len(gallery_columns)]:
                st.image(
                    photo["content"],
                    caption=photo.get("filename", f"Property photo {index + 1}"),
                    use_container_width=True,
                )
        preview_options = [
            f"{index + 1}. {photo.get('filename', 'Property photo')}"
            for index, photo in enumerate(preview_photos)
        ]
        selected_preview = st.selectbox(
            "Choose a photo for the reveal comparison",
            preview_options,
            key="photo_preview_selection",
        )
        preview_index = preview_options.index(selected_preview)
        preview_analysis = (
            preview_analyses[preview_index]
            if preview_index < len(preview_analyses)
            else {}
        )
        quality_score = preview_analysis.get("quality_score")
        if quality_score is None:
            try:
                quality_score = cached_photo_quality(
                    preview_photos[preview_index]["content"]
                )["quality_score"]
            except (OSError, ValueError) as error:
                st.warning(f"Could not rate this photo's image quality: {error}")
        render_image_comparison(
            preview_photos[preview_index]["content"],
            preview_analysis.get("room_type"),
            quality_score,
            preview_index,
        )

with ai_studio_tab:
    st.subheader("AI property listing & renovation studio")
    last_valuation = st.session_state.get("last_valuation")
    if not last_valuation:
        st.info("Run a home valuation first to prepare listing copy and renovation scenarios.")
    else:
        listing_attributes = last_valuation["attributes"]
        st.markdown("#### Listing & marketing copy")
        st.caption(
            "Uses the configured OpenAI API only after you click generate. "
            "Property details below are sent to that API; no photo is sent."
        )
        if listing_attributes.get("Property type") != "Home":
            st.info("Marketing copy is currently configured for residential homes.")
        else:
            if st.button(
                "Generate property listing",
                type="primary",
                key="generate_ai_listing",
            ):
                st.session_state.pop("generated_ai_listing", None)
                listing_inputs = {
                    "living_area_sqft": listing_attributes.get("Living area (sq.ft)"),
                    "bedrooms": listing_attributes.get("Bedrooms"),
                    "bathrooms": listing_attributes.get("Bathrooms"),
                    "age_years": listing_attributes.get("Age (years)"),
                    "locality": listing_attributes.get("Locality"),
                    "visual_quality_score": last_valuation.get("visual_quality_score"),
                    "finish_lighting_score": listing_attributes.get(
                        "Property finish & lighting score"
                    ),
                    "location_score": listing_attributes.get("Location score"),
                }
                try:
                    generated_listing = generate_property_listing(listing_inputs)
                except (RuntimeError, ValueError) as error:
                    st.error(f"Could not generate listing copy: {error}")
                else:
                    st.session_state.generated_ai_listing = generated_listing
            generated_listing = st.session_state.get("generated_ai_listing")
            if generated_listing:
                st.markdown(f"### {generated_listing['headline']}")
                st.markdown("**Key features**")
                for feature in generated_listing["key_features"]:
                    st.markdown(f"- {feature}")
                st.markdown("**Property description**")
                st.write(generated_listing["description"])

        st.divider()
        st.markdown("#### Renovation what-if simulator")
        st.caption(
            "Indicative India-market cost ranges; model deltas are scenario estimates, "
            "not guaranteed renovation returns."
        )
        upgrades = st.multiselect(
            "Select virtual upgrades",
            [
                "Add one bathroom",
                "Kitchen renovation",
                "Modern flooring",
                "Solar panels",
            ],
            key="renovation_upgrades",
        )
        if not upgrades:
            st.info("Choose upgrades to re-run the valuation model and estimate ROI.")
        elif usd_inr_rate is None:
            st.error("Renovation ROI needs the current USD-to-INR rate.")
        elif not last_valuation.get("model_features"):
            st.info("Run a fresh home valuation to attach model inputs for this scenario.")
        elif listing_attributes.get("Property type") != "Home":
            st.info("The renovation simulator applies to the residential home model.")
        else:
            living_area = float(last_valuation["model_features"].get("sqft_living", 0))
            scenario_features = dict(last_valuation["model_features"])
            unmodeled_upgrades = []
            if "Add one bathroom" in upgrades:
                scenario_features["bathrooms"] = (
                    float(scenario_features.get("bathrooms", 0)) + 1
                )
            if any(
                upgrade in upgrades
                for upgrade in ("Kitchen renovation", "Modern flooring")
            ):
                if "is_renovated" in model_features:
                    scenario_features["is_renovated"] = 1.0
                else:
                    unmodeled_upgrades.extend(
                        upgrade
                        for upgrade in upgrades
                        if upgrade in ("Kitchen renovation", "Modern flooring")
                    )
            if "Solar panels" in upgrades:
                unmodeled_upgrades.append("Solar panels")

            try:
                scenario_data = pd.DataFrame(
                    [
                        {
                            column: float(scenario_features.get(column, 0.0))
                            for column in model_features
                        }
                    ],
                    columns=model_features,
                )
                scenario_value = float(model.predict(scenario_data)[0])
                if not math.isfinite(scenario_value):
                    raise ValueError("The model returned an invalid scenario estimate.")
                cost_ranges = [
                    renovation_cost_range(upgrade, living_area)
                    for upgrade in upgrades
                ]
            except (TypeError, ValueError, RuntimeError) as error:
                st.error(f"Could not calculate the renovation scenario: {error}")
            else:
                total_cost_low = sum(cost[0] for cost in cost_ranges)
                total_cost_high = sum(cost[1] for cost in cost_ranges)
                value_bump_inr = (
                    scenario_value - float(last_valuation["predicted_value"])
                ) * usd_inr_rate
                roi_at_high_cost = (
                    (value_bump_inr - total_cost_high) / total_cost_high * 100
                    if total_cost_high > 0
                    else 0.0
                )
                roi_at_low_cost = (
                    (value_bump_inr - total_cost_low) / total_cost_low * 100
                    if total_cost_low > 0
                    else 0.0
                )
                result_columns = st.columns(3)
                result_columns[0].metric(
                    "Model-estimated value bump",
                    format_indian_rupees(value_bump_inr),
                )
                result_columns[1].metric(
                    "Indicative renovation cost",
                    f"{format_indian_rupees(total_cost_low)}–"
                    f"{format_indian_rupees(total_cost_high)}",
                )
                result_columns[2].metric(
                    "Projected ROI range",
                    f"{roi_at_high_cost:.1f}% to {roi_at_low_cost:.1f}%",
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Upgrade": upgrade,
                                "Estimated cost low (INR)": cost[0],
                                "Estimated cost high (INR)": cost[1],
                            }
                            for upgrade, cost in zip(upgrades, cost_ranges)
                        ]
                    ).style.format(
                        {
                            "Estimated cost low (INR)": format_indian_rupees,
                            "Estimated cost high (INR)": format_indian_rupees,
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if unmodeled_upgrades:
                    st.warning(
                        "The model has no dedicated energy-efficiency or solar input. "
                        "Solar panels do not add a learned price premium here. Kitchen "
                        "and flooring are represented only by the training dataset's "
                        "general is_renovated flag when available. Unmodeled items "
                        "therefore have no modeled uplift."
                    )
                st.caption(
                    "Model scenario changes are limited to the bathroom count and, "
                    "where supported, the general renovation flag. Feature effects "
                    "are learned associations, not causal renovation appraisals. "
                    "Cost ranges are planning estimates and vary by city and scope."
                )

with report_tab:
    st.markdown("**Nearby amenities (OpenStreetMap)**")
    st.caption(
        "Searches up to 1.5 km from the selected map coordinates. Scores reflect "
        "the nearest mapped feature only and depend on OSM coverage."
    )
    amenity_latitude = float(st.session_state.selected_lat)
    amenity_longitude = float(st.session_state.selected_lon)
    if st.button("Check nearby amenities", key="check_nearby_amenities"):
        try:
            amenity_results = cached_nearby_amenities(
                amenity_latitude,
                amenity_longitude,
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            st.session_state.amenity_lookup = {
                "latitude": amenity_latitude,
                "longitude": amenity_longitude,
                "results": amenity_results,
            }
    amenity_lookup = st.session_state.get("amenity_lookup")
    if (
        amenity_lookup
        and amenity_lookup["latitude"] == amenity_latitude
        and amenity_lookup["longitude"] == amenity_longitude
    ):
        amenity_columns = st.columns(2)
        for index, (category, details) in enumerate(
            amenity_lookup["results"].items()
        ):
            with amenity_columns[index % len(amenity_columns)]:
                nearest = details["nearest_distance_m"]
                distance_label = (
                    f"{nearest:.0f} m away" if nearest is not None else "Not mapped nearby"
                )
                st.markdown(
                    f"**{category}** · {details['count']} found · {distance_label}"
                )
                st.progress(int(details["score"]), text=f"Proximity score: {details['score']}/100")
        st.caption("Amenity data © OpenStreetMap contributors; queried via Overpass API.")
        render_amenity_radar(
            amenity_lookup["results"],
            key="micro_location_amenity_radar",
        )

    st.markdown("**Nearby historical sales comparables**")
    st.caption(
        "The top three feature-similar records from the local historical dataset "
        "within 2 km are shown. These are not live or independently verified "
        "market transactions."
    )
    comparable_valuation = st.session_state.get("last_valuation")
    comparable_attributes = (
        comparable_valuation.get("attributes", {})
        if comparable_valuation
        else {}
    )
    try:
        comparable_features = {
            "living_area": float(
                str(
                    comparable_attributes.get(
                        "Living area (sq.ft)",
                        area,
                    )
                ).replace(",", "")
            ),
            "bedrooms": float(
                comparable_attributes.get("Bedrooms", bedrooms)
            ),
            "bathrooms": float(
                comparable_attributes.get("Bathrooms", bathrooms)
            ),
            "age": float(
                comparable_attributes.get("Age (years)", age)
            ),
        }
        comparable_latitude = float(
            comparable_valuation.get("latitude", amenity_latitude)
            if comparable_valuation
            else amenity_latitude
        )
        comparable_longitude = float(
            comparable_valuation.get("longitude", amenity_longitude)
            if comparable_valuation
            else amenity_longitude
        )
    except (TypeError, ValueError, OverflowError) as error:
        st.error(f"Could not read the current property features for comparisons: {error}")
    else:
        excluded_property_id = comparable_attributes.get("Property ID")
        if excluded_property_id == "Not available":
            excluded_property_id = None
        try:
            comparables = nearest_historical_comparables(
                source_data,
                comparable_latitude,
                comparable_longitude,
                living_area=comparable_features["living_area"],
                bedrooms=comparable_features["bedrooms"],
                bathrooms=comparable_features["bathrooms"],
                age=comparable_features["age"],
                excluded_property_id=excluded_property_id,
            )
        except ValueError as error:
            st.error(f"Could not prepare historical comparables: {error}")
        else:
            if comparables.empty:
                st.info(
                    "No valid historical sales records were found within 2 km "
                    "of this property's location."
                )
            else:
                comp_columns = st.columns(len(comparables))
                for index, (_, comparable) in enumerate(comparables.iterrows()):
                    with comp_columns[index].container(border=True):
                        property_id = normalize_property_id(comparable["id"])
                        st.markdown(f"**Comparable {index + 1} · {property_id}**")
                        st.metric(
                            "Similarity",
                            f"{comparable['similarity_score']:.0f}%",
                        )
                        st.metric(
                            "Recorded sale price (USD)",
                            f"${comparable['price']:,.0f}",
                        )
                        if usd_inr_rate is not None:
                            st.caption(
                                "Converted: "
                                + format_indian_rupees(
                                    comparable["price"] * usd_inr_rate
                                )
                            )
                        st.metric(
                            "Distance",
                            f"{comparable['distance_m'] / 1000:.2f} km",
                        )
                        st.metric(
                            "Living area",
                            f"{comparable['sqft_living']:,.0f} sq.ft",
                        )
                        st.caption(
                            f"{comparable.get('area_name', 'Local record')}"
                        )

                if usd_inr_rate is None:
                    st.warning(
                        "The USD-to-INR service is unavailable; comparable prices "
                        "are shown in their recorded USD currency."
                    )
                else:
                    comps_map = make_comparables_map(
                        comparables,
                        comparable_latitude,
                        comparable_longitude,
                        usd_inr_rate,
                    )
                    st_folium(
                        comps_map,
                        height=420,
                        use_container_width=True,
                        returned_objects=[],
                        key="historical_comparables_map",
                    )

with market_tab:
    st.subheader("Historical dataset snapshot")
    st.caption(
        "This chart summarizes the local training dataset, not live market data. "
        "Only years present in the dataset are shown. Historical prices are "
        "converted using the latest published exchange rate, not historical rates."
    )
    if "year" in source_data and "price" in source_data:
        trend_data = source_data[["year", "price"]].copy()
        trend_data["year"] = pd.to_numeric(trend_data["year"], errors="coerce")
        trend_data["price"] = pd.to_numeric(trend_data["price"], errors="coerce")
        if "area_name" in source_data:
            trend_data["area_name"] = source_data["area_name"]
            locality_options = ["All localities"] + sorted(
                trend_data["area_name"].dropna().astype(str).unique().tolist()
            )
            selected_market_area = st.selectbox(
                "Locality",
                locality_options,
                key="market_trends_area",
            )
            if selected_market_area != "All localities":
                trend_data = trend_data[
                    trend_data["area_name"] == selected_market_area
                ]
        trend_data = trend_data.dropna(subset=["year", "price"])
        if trend_data.empty:
            st.info("There are no dated price records for this locality.")
        elif usd_inr_rate is None:
            st.error("INR market-trend values are unavailable without the exchange rate.")
        else:
            median_by_year = (
                trend_data.groupby("year", as_index=False)["price"]
                .median()
                .sort_values("year")
            )
            median_by_year["price"] *= usd_inr_rate
            chart_data = median_by_year.rename(
                columns={
                    "year": "Year",
                    "price": "Median recorded price (INR)",
                }
            )
            st.line_chart(
                chart_data,
                x="Year",
                y="Median recorded price (INR)",
            )
            st.dataframe(
                chart_data.style.format(
                    {
                        "Median recorded price (INR)": lambda value: (
                            format_indian_rupees(value)
                        )
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            if median_by_year["year"].nunique() < 2:
                st.caption(
                    "Only one year is available for this selection; no price trend "
                    "can be inferred."
                )
    else:
        st.info("This dataset has no year and price fields for a historical snapshot.")

with investment_tab:
    st.subheader("Rental yield and financing")
    last_valuation = st.session_state.get("last_valuation")
    if not last_valuation:
        st.info("Run a property valuation first; ROI estimates use its predicted value.")
    elif usd_inr_rate is None:
        st.error("INR investment estimates are unavailable without the exchange rate.")
    else:
        st.caption(
            "Planning scenario only. Adjust the assumptions; projected rent is "
            "calculated from the editable annual gross yield assumption."
        )
        rent_yield = st.slider(
            "Expected annual gross rent yield (%)",
            min_value=1.0,
            max_value=15.0,
            value=4.0,
            step=0.25,
            help="Editable planning assumption; this is not an observed rent quote.",
            key="investment_rent_yield",
        )
        annual_appreciation = st.slider(
            "Expected annual property appreciation (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.25,
            help="Planning assumption used to estimate future property value and investor equity.",
            key="investment_annual_appreciation",
        )
        finance_col1, finance_col2 = st.columns(2)
        with finance_col1:
            down_payment_pct = st.slider(
                "Down payment (%)",
                min_value=0,
                max_value=100,
                value=20,
                help="Share of the estimated property value paid upfront.",
                key="investment_down_payment",
            )
            annual_interest_pct = st.slider(
                "Annual interest rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.25,
                key="investment_interest_rate",
            )
        with finance_col2:
            loan_term_years = st.slider(
                "Loan tenure (years)",
                min_value=1,
                max_value=40,
                value=20,
                key="investment_loan_term",
            )
            monthly_maintenance = st.slider(
                "Expected monthly maintenance (INR)",
                min_value=0,
                max_value=500000,
                value=20000,
                step=5000,
                key="investment_maintenance",
            )
        try:
            investment_metrics = calculate_investment_metrics(
                property_value=float(last_valuation["predicted_value"]),
                annual_rent_yield_pct=float(rent_yield),
                down_payment_pct=float(down_payment_pct),
                annual_interest_pct=float(annual_interest_pct),
                loan_term_years=int(loan_term_years),
                monthly_maintenance=float(monthly_maintenance) / usd_inr_rate,
            )
            cash_flow_projection = project_investment_cash_flows(
                property_value=float(last_valuation["predicted_value"]),
                annual_rent_yield_pct=float(rent_yield),
                down_payment_pct=float(down_payment_pct),
                annual_interest_pct=float(annual_interest_pct),
                loan_term_years=int(loan_term_years),
                annual_appreciation_pct=float(annual_appreciation),
                monthly_maintenance=float(monthly_maintenance) / usd_inr_rate,
            )
        except ValueError as error:
            st.error(f"Could not calculate investment estimates: {error}")
        else:
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Estimated monthly rent",
                format_indian_rupees(investment_metrics["monthly_rent"] * usd_inr_rate),
            )
            metric_columns[1].metric(
                "Annual gross rental income",
                format_indian_rupees(
                    investment_metrics["annual_gross_rent"] * usd_inr_rate
                ),
            )
            metric_columns[2].metric(
                "Estimated cap rate",
                f"{investment_metrics['cap_rate_pct']:.2f}%",
            )
            detail_columns = st.columns(3)
            detail_columns[0].metric(
                "Estimated monthly mortgage",
                format_indian_rupees(
                    investment_metrics["monthly_mortgage"] * usd_inr_rate
                ),
            )
            detail_columns[1].metric(
                "Monthly cash flow after loan & maintenance",
                format_indian_rupees(
                    investment_metrics["monthly_cash_flow"] * usd_inr_rate
                ),
            )
            detail_columns[2].metric(
                "Down payment",
                format_indian_rupees(
                    investment_metrics["down_payment"] * usd_inr_rate
                ),
            )
            st.caption(
                "Cap rate = (annual gross rent − annual maintenance) ÷ property "
                "value. Amounts are displayed in INR using the published USD-to-INR "
                "rate. Mortgage payments exclude taxes, insurance, fees, vacancy, "
                "and other operating costs; actual financing and rent will vary."
            )
            st.markdown("**10-year cash flow and equity projection**")
            st.caption(
                "Projection uses the selected rent yield and appreciation assumptions. "
                "ROI includes cumulative operating cash flow and estimated equity growth "
                "relative to the initial down payment. Rent is estimated at the selected "
                "yield on projected property value; taxes, transaction costs, vacancy, "
                "and other costs are not modeled."
            )
            projection_table = pd.DataFrame(cash_flow_projection).rename(
                columns={
                    "year": "Year",
                    "gross_rent": "Gross rent (INR)",
                    "maintenance": "Maintenance (INR)",
                    "debt_service": "Debt service (INR)",
                    "net_cash_flow": "Net cash flow (INR)",
                    "cumulative_cash_flow": "Cumulative cash flow (INR)",
                    "property_value": "Estimated property value (INR)",
                    "remaining_loan": "Remaining loan (INR)",
                    "equity": "Estimated equity (INR)",
                    "roi_pct": "ROI (%)",
                }
            )
            currency_columns = [
                column for column in projection_table.columns
                if column.endswith("(INR)")
            ]
            projection_table[currency_columns] *= usd_inr_rate
            st.dataframe(
                projection_table.style.format(
                    {
                        column: lambda value: format_indian_rupees(value)
                        for column in currency_columns
                    } | {"ROI (%)": lambda value: f"{value:.1f}%"},
                    na_rep="N/A",
                ),
                hide_index=True,
                use_container_width=True,
            )
            if down_payment_pct > 0:
                roi_chart_data = pd.DataFrame(
                    {
                        "Year": [row["year"] for row in cash_flow_projection],
                        "ROI (%)": [row["roi_pct"] for row in cash_flow_projection],
                    }
                )
                roi_chart = px.line(
                    roi_chart_data,
                    x="Year",
                    y="ROI (%)",
                    markers=True,
                    title="Projected 10-year investor ROI",
                )
                roi_chart.update_traces(
                    hovertemplate="Year %{x}<br>ROI %{y:.1f}%<extra></extra>"
                )
                roi_chart.update_layout(
                    yaxis_title="ROI on initial down payment",
                    xaxis_dtick=1,
                    hovermode="x unified",
                )
                roi_chart.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(roi_chart, use_container_width=True)
            else:
                st.info(
                    "ROI relative to an initial down payment is undefined when the "
                    "down payment is 0%; the 10-year cash-flow projection is still shown."
                )

with inspection_tab:
    st.subheader("Inspection logs")
    saved_records = list_inspections()
    if not saved_records:
        st.info("No inspection sessions have been saved yet.")
    else:
        record_options = {
            (
                f"{record['created_at']} · {record['asset_type']} · "
                f"{record['location']} · {record['inspection_id'][:8]}"
            ): record
            for record in saved_records
        }
        selected_record_label = st.selectbox(
            "Inspection session",
            options=list(record_options),
            key="tab_inspection_record",
        )
        selected_record = record_options[selected_record_label]
        st.caption(
            f"{selected_record['asset_type']} · {selected_record['location']} · "
            f"{len(selected_record['photos'])} saved photo(s)"
        )
        if selected_record["photos"]:
            saved_photo_columns = st.columns(
                min(3, len(selected_record["photos"]))
            )
            session_dir = DEFAULT_INSPECTIONS_DIR / selected_record["inspection_id"]
            for photo_index, photo in enumerate(selected_record["photos"]):
                with saved_photo_columns[photo_index % len(saved_photo_columns)]:
                    st.image(
                        str(session_dir / photo["file"]),
                        caption=photo["category"],
                        use_container_width=True,
                    )
        st.download_button(
            "Download inspection metadata",
            data=json.dumps(selected_record, indent=2, ensure_ascii=False),
            file_name=f"inspection-{selected_record['inspection_id']}.json",
            mime="application/json",
            key="tab_inspection_download",
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI-Based Real Estate Valuation • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)