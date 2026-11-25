"""Configurable RapidAPI adapter for live real-estate listings.

The API host, path, response shape, and query parameters vary by marketplace.
Configure ``RAPIDAPI_HOST``, ``RAPIDAPI_KEY``, and ``RAPIDAPI_LISTINGS_PATH``.
The default query uses latitude, longitude, city, radius_km, and search_type.
Override query names or add provider-specific parameters with the JSON
environment variables documented in this module's function docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class PropertyListingServiceError(RuntimeError):
    """Raised when the configured live listing API cannot return valid results."""


@dataclass(frozen=True)
class PropertyListing:
    """Normalized property listing from a configured external API."""

    listing_id: str
    title: str
    address: str
    price: float | None
    currency: str
    bedrooms: float | None
    bathrooms: float | None
    square_feet: float | None
    listing_url: str
    image_url: str
    latitude: float
    longitude: float
    distance_km: float
    search_type: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view of the normalized listing."""
        return {
            "listing_id": self.listing_id,
            "title": self.title,
            "address": self.address,
            "price": self.price,
            "currency": self.currency,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "square_feet": self.square_feet,
            "listing_url": self.listing_url,
            "image_url": self.image_url,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "distance_km": self.distance_km,
            "search_type": self.search_type,
            "source": self.source,
        }


DEFAULT_QUERY_PARAM_NAMES = {
    "latitude": "latitude",
    "longitude": "longitude",
    "city": "city",
    "radius_km": "radius_km",
    "search_type": "search_type",
}

LISTING_COLLECTION_KEYS = (
    "listings",
    "properties",
    "results",
    "homes",
    "items",
    "search_results",
    "searchResults",
    "data",
)


def is_configured() -> bool:
    """Return whether required RapidAPI host and key settings are present."""
    return bool(
        os.environ.get("RAPIDAPI_HOST", "").strip()
        and os.environ.get("RAPIDAPI_KEY", "").strip()
    )


def _environment_json(name: str, default: dict[str, str]) -> dict[str, str]:
    raw_value = os.environ.get(name)
    if not raw_value:
        return dict(default)
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain a valid JSON object.") from error
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in decoded.items()
    ):
        raise ValueError(f"{name} must contain a JSON object of string values.")
    return decoded


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        match = re.search(r"-?\d+(?:[,.]\d+)*(?:\.\d+)?", str(value).replace(",", ""))
        if not match:
            return None
        try:
            parsed = float(match.group(0))
        except ValueError:
            return None
    return parsed if math.isfinite(parsed) else None


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _coordinates(record: dict[str, Any]) -> tuple[float | None, float | None]:
    latitude = _number(
        _first(record, ("latitude", "lat", "listing_latitude", "y"))
    )
    longitude = _number(
        _first(record, ("longitude", "lon", "lng", "listing_longitude", "x"))
    )
    location = record.get("location")
    coordinates = record.get("coordinates")
    for nested in (location, coordinates):
        if isinstance(nested, dict):
            latitude = latitude if latitude is not None else _number(
                _first(nested, ("latitude", "lat"))
            )
            longitude = longitude if longitude is not None else _number(
                _first(nested, ("longitude", "lon", "lng"))
            )
        elif isinstance(nested, (list, tuple)) and len(nested) >= 2:
            # GeoJSON coordinate arrays are [longitude, latitude].
            if longitude is None:
                longitude = _number(nested[0])
            if latitude is None:
                latitude = _number(nested[1])
    return latitude, longitude


def _collection(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise PropertyListingServiceError("The listing API returned an invalid JSON shape.")
    for key in LISTING_COLLECTION_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _collection(value)
            if nested:
                return nested
    raise PropertyListingServiceError(
        "The listing API response contains no recognized listing collection. "
        "Configure its response adapter before using live results."
    )


def _haversine_km(
    latitude: float,
    longitude: float,
    other_latitude: float,
    other_longitude: float,
) -> float:
    earth_radius_km = 6371.0088
    lat1, lat2 = math.radians(latitude), math.radians(other_latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(other_longitude - longitude)
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(
        min(1.0, math.sqrt(haversine))
    )


def _normalize_listing(
    raw: dict[str, Any],
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
    search_type: str,
    source: str,
    default_currency: str,
) -> PropertyListing | None:
    listing_latitude, listing_longitude = _coordinates(raw)
    if (
        listing_latitude is None
        or listing_longitude is None
        or not -90 <= listing_latitude <= 90
        or not -180 <= listing_longitude <= 180
    ):
        return None
    distance_km = _haversine_km(
        latitude,
        longitude,
        listing_latitude,
        listing_longitude,
    )
    if distance_km > radius_km:
        return None

    raw_address = raw.get("address")
    address = (
        " ".join(
            str(raw_address.get(key, "")).strip()
            for key in ("street", "city", "state", "postal_code")
            if raw_address.get(key)
        )
        if isinstance(raw_address, dict)
        else str(raw_address or raw.get("formatted_address") or "Address unavailable")
    )
    price_value = _first(raw, ("price", "list_price", "asking_price", "rent"))
    currency_value = str(
        _first(raw, ("currency", "price_currency")) or default_currency
    ).upper()
    if isinstance(price_value, dict):
        currency_value = str(
            price_value.get("currency", currency_value)
        ).upper()
        price_value = _first(price_value, ("amount", "value", "raw"))
    title = str(
        _first(raw, ("title", "name", "property_name", "description"))
        or address
    ).strip()
    listing_url = str(
        _first(raw, ("listing_url", "url", "detail_url", "permalink"))
        or ""
    ).strip()
    image_value = _first(
        raw,
        ("image_url", "thumbnail", "thumbnail_url", "photo", "primary_photo"),
    )
    if isinstance(image_value, dict):
        image_value = _first(image_value, ("url", "src", "href"))

    return PropertyListing(
        listing_id=str(
            _first(raw, ("id", "listing_id", "property_id", "mls_id"))
            or f"{listing_latitude:.6f},{listing_longitude:.6f}"
        ),
        title=title,
        address=address,
        price=_number(price_value),
        currency=currency_value,
        bedrooms=_number(_first(raw, ("bedrooms", "beds", "beds_count"))),
        bathrooms=_number(_first(raw, ("bathrooms", "baths", "baths_count"))),
        square_feet=_number(
            _first(raw, ("square_feet", "sqft", "living_area", "area"))
        ),
        listing_url=listing_url,
        image_url=str(image_value or "").strip(),
        latitude=listing_latitude,
        longitude=listing_longitude,
        distance_km=distance_km,
        search_type=search_type,
        source=source,
    )


def fetch_live_property_listings(
    *,
    latitude: float,
    longitude: float,
    city: str,
    radius_km: float,
    search_type: str,
    host: str | None = None,
    api_key: str | None = None,
    search_path: str | None = None,
    timeout_seconds: int = 20,
) -> list[dict[str, Any]]:
    """Fetch and normalize listings from a configured RapidAPI-compatible host.

    Configuration:
    - ``RAPIDAPI_HOST`` and ``RAPIDAPI_KEY`` (required).
    - ``RAPIDAPI_LISTINGS_PATH`` (defaults to ``/search``).
    - ``RAPIDAPI_QUERY_PARAM_NAMES_JSON`` can rename default query keys, e.g.
      ``{"latitude":"lat","longitude":"lon","radius_km":"radius"}``.
    - ``RAPIDAPI_EXTRA_QUERY_PARAMS_JSON`` adds provider-specific string query
      parameters, e.g. ``{"property_type":"residential"}``.
    - ``RAPIDAPI_DEFAULT_CURRENCY`` labels results if the provider omits currency.

    A marketplace's host, endpoint, query semantics, and response fields are
    provider-specific; set the path and query parameters for that API. Responses
    should include a recognized listings array and latitude/longitude per item.
    """
    if not all(
        math.isfinite(value)
        for value in (latitude, longitude, radius_km)
    ):
        raise ValueError("Coordinates and search radius must be finite numbers.")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude or longitude is outside its valid range.")
    if not 0.5 <= radius_km <= 10.0:
        raise ValueError("Search radius must be between 0.5 and 10 km.")
    normalized_search_type = search_type.strip().lower()
    if normalized_search_type not in {"for sale", "for rent"}:
        raise ValueError("Search type must be 'For Sale' or 'For Rent'.")

    configured_host = (host or os.environ.get("RAPIDAPI_HOST", "")).strip()
    configured_key = (api_key or os.environ.get("RAPIDAPI_KEY", "")).strip()
    if not configured_host or not configured_key:
        raise PropertyListingServiceError(
            "Live listings are not configured. Set RAPIDAPI_HOST and RAPIDAPI_KEY."
        )
    configured_host = configured_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    configured_path = (
        search_path
        or os.environ.get("RAPIDAPI_LISTINGS_PATH", "/search")
    ).strip()
    if not configured_path.startswith("/"):
        configured_path = f"/{configured_path}"

    parameter_names = _environment_json(
        "RAPIDAPI_QUERY_PARAM_NAMES_JSON",
        DEFAULT_QUERY_PARAM_NAMES,
    )
    query_values = {
        parameter_names.get("latitude", "latitude"): str(latitude),
        parameter_names.get("longitude", "longitude"): str(longitude),
        parameter_names.get("city", "city"): city.strip(),
        parameter_names.get("radius_km", "radius_km"): str(radius_km),
        parameter_names.get("search_type", "search_type"): normalized_search_type,
    }
    query_values.update(
        _environment_json("RAPIDAPI_EXTRA_QUERY_PARAMS_JSON", {})
    )
    query_values = {key: value for key, value in query_values.items() if value != ""}
    request = Request(
        f"https://{configured_host}{configured_path}?{urlencode(query_values)}",
        headers={
            "X-RapidAPI-Key": configured_key,
            "X-RapidAPI-Host": configured_host,
            "Accept": "application/json",
            "User-Agent": "AI-Estate/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        status_message = {
            401: "RapidAPI rejected the configured key.",
            403: "RapidAPI denied access to this API or subscription plan.",
            429: "The live-listing API rate limit was reached.",
        }.get(error.code, f"The live-listing API returned HTTP {error.code}.")
        raise PropertyListingServiceError(status_message) from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise PropertyListingServiceError(
            f"Could not retrieve live listings: {error}"
        ) from error

    raw_listings = _collection(payload)
    source = configured_host
    default_currency = os.environ.get("RAPIDAPI_DEFAULT_CURRENCY", "INR").upper()
    normalized = []
    for raw_listing in raw_listings:
        listing = _normalize_listing(
            raw_listing,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            search_type=search_type,
            source=source,
            default_currency=default_currency,
        )
        if listing is not None:
            normalized.append(listing.as_dict())
    return sorted(normalized, key=lambda item: item["distance_km"])
