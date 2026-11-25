"""
Broker Portal and Real-Time Property Listing Management Module.
Provides persistent storage, live Leaflet location tagging, camera/photo capture,
AI-powered photo quality inspection, listing copy generation, and main dashboard integration.
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from vision_module import assess_photo_quality

BROKER_DATA_DIR = Path(__file__).resolve().parent / "broker_data"
LISTINGS_FILE = BROKER_DATA_DIR / "listings.json"
PHOTOS_DIR = BROKER_DATA_DIR / "photos"


def init_broker_storage():
    """Ensure broker data directories and index file exist with sample seed data."""
    BROKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    if not LISTINGS_FILE.exists():
        # Populate with initial high-quality broker listings in major tech hubs
        sample_listings = [
            {
                "listing_id": "BRK-2026-BLR01",
                "title": "Luxury 3 BHK Sky Villa with Private Deck",
                "property_type": "Apartment / Flat",
                "listing_type": "For Sale",
                "price_inr": 18500000,
                "bedrooms": 3,
                "bathrooms": 3,
                "balconies": 2,
                "area_sqft": 2150.0,
                "furnishing": "Semi-Furnished",
                "floor": 14,
                "total_floors": 24,
                "property_age": 2,
                "amenities": ["Covered Parking", "24/7 Security", "Power Backup", "Gym", "Swimming Pool", "Club House", "Children Play Area"],
                "latitude": 12.9352,
                "longitude": 77.6245,
                "address": "4th Block, 80 Feet Road, Koramangala",
                "city": "Bengaluru",
                "state": "Karnataka",
                "pincode": "560034",
                "broker_name": "Vikram Malhotra",
                "broker_phone": "+91 98450 12345",
                "broker_email": "vikram@primeestates.in",
                "broker_agency": "Prime Realty Advisors",
                "rera_id": "PRM/KA/RERA/1251/310/AG/210423/001890",
                "status": "Active",
                "photo_files": ["sample_broker_1.jpg"],
                "primary_photo": "sample_broker_1.jpg",
                "photo_quality_score": 88.5,
                "photo_stars": 5,
                "headline": "Premier 3 BHK Sky Villa in the Heart of Koramangala",
                "description": "Breathtaking panoramic urban views from a private 14th-floor deck. Premium Italian marble flooring, German fittings, EV charging bay, and exclusive clubhouse access.",
                "created_at": "2026-11-15T10:30:00Z",
                "views_count": 42,
            },
            {
                "listing_id": "BRK-2026-BLR02",
                "title": "Contemporary 2 BHK Garden Apartment",
                "property_type": "Apartment / Flat",
                "listing_type": "For Sale",
                "price_inr": 9200000,
                "bedrooms": 2,
                "bathrooms": 2,
                "balconies": 1,
                "area_sqft": 1280.0,
                "furnishing": "Furnished",
                "floor": 3,
                "total_floors": 12,
                "property_age": 4,
                "amenities": ["Covered Parking", "24/7 Security", "Power Backup", "Gym", "Lift", "Jogging Track"],
                "latitude": 12.9784,
                "longitude": 77.6408,
                "address": "12th Main Road, HAL 2nd Stage, Indiranagar",
                "city": "Bengaluru",
                "state": "Karnataka",
                "pincode": "560038",
                "broker_name": "Pooja Reddy",
                "broker_phone": "+91 98860 67890",
                "broker_email": "pooja.reddy@metroliving.co",
                "broker_agency": "Metro Living Properties",
                "rera_id": "PRM/KA/RERA/1251/310/AG/220511/002341",
                "status": "Active",
                "photo_files": ["sample_broker_2.jpg"],
                "primary_photo": "sample_broker_2.jpg",
                "photo_quality_score": 82.0,
                "photo_stars": 4,
                "headline": "Fully Furnished Urban Oasis near Indiranagar Metro",
                "description": "Sunlit east-facing apartment with modern modular kitchen, acoustic double-glazed windows, and walking distance to 100 Feet Road restaurants and cafes.",
                "created_at": "2026-11-16T14:15:00Z",
                "views_count": 28,
            },
            {
                "listing_id": "BRK-2026-MYS01",
                "title": "Independent 4 BHK Heritage Villa with Private Lawn",
                "property_type": "Independent House / Villa",
                "listing_type": "For Sale",
                "price_inr": 14500000,
                "bedrooms": 4,
                "bathrooms": 4,
                "balconies": 3,
                "area_sqft": 2800.0,
                "furnishing": "Semi-Furnished",
                "floor": 1,
                "total_floors": 2,
                "property_age": 5,
                "amenities": ["Covered Parking", "Private Garden", "Solar Water Heating", "Rainwater Harvesting", "Security", "Vaastu Compliant"],
                "latitude": 12.2866,
                "longitude": 76.6376,
                "address": "Kuvempunagar 5th Cross, Near Complex",
                "city": "Mysuru",
                "state": "Karnataka",
                "pincode": "570023",
                "broker_name": "Suresh Gowda",
                "broker_phone": "+91 94480 54321",
                "broker_email": "suresh@mysoreheritagehomes.com",
                "broker_agency": "Heritage City Realty",
                "rera_id": "PRM/KA/RERA/1251/310/AG/200315/001120",
                "status": "Active",
                "photo_files": ["sample_broker_3.jpg"],
                "primary_photo": "sample_broker_3.jpg",
                "photo_quality_score": 85.0,
                "photo_stars": 4,
                "headline": "Spacious Vaastu-Compliant 4 BHK Villa in Peaceful Kuvempunagar",
                "description": "Architect-designed independent villa featuring teakwood doors, expansive landscaped courtyard, two car parks, and tranquil residential surroundings.",
                "created_at": "2026-11-16T18:00:00Z",
                "views_count": 19,
            }
        ]
        LISTINGS_FILE.write_text(json.dumps(sample_listings, indent=2, ensure_ascii=False), encoding="utf-8")


# Initialize upon module import
init_broker_storage()


def get_all_broker_listings(status: str | None = None) -> list[dict]:
    """Return all broker listings, optionally filtered by status, sorted newest first."""
    init_broker_storage()
    try:
        data = json.loads(LISTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        if status:
            data = [item for item in data if item.get("status") == status]
        return sorted(data, key=lambda x: x.get("created_at", ""), reverse=True)
    except (OSError, json.JSONDecodeError):
        return []


def get_broker_listing(listing_id: str) -> dict | None:
    """Retrieve a single broker listing by its ID."""
    listings = get_all_broker_listings()
    for item in listings:
        if item.get("listing_id") == listing_id:
            return item
    return None


def save_broker_listing(
    metadata: dict[str, Any],
    photos: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Persist a newly submitted broker property listing with photos and AI scores.
    Each photo entry in photos should be: {"filename": str, "bytes": bytes, "is_primary": bool}
    """
    init_broker_storage()
    now_utc = datetime.now(timezone.utc)
    listing_id = f"BRK-{now_utc.strftime('%Y%m%d')}-{uuid4().hex[:6].upper()}"

    listing_photos_dir = PHOTOS_DIR / listing_id
    listing_photos_dir.mkdir(parents=True, exist_ok=True)

    saved_photo_names = []
    primary_photo_name = None
    photo_quality_scores = []

    for idx, photo in enumerate(photos, start=1):
        content = photo.get("bytes")
        if not content:
            continue
        orig_name = photo.get("filename", f"photo_{idx}.jpg")
        ext = Path(orig_name).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        save_name = f"property_photo_{idx:02d}{ext}"
        dest_path = listing_photos_dir / save_name
        dest_path.write_bytes(content)
        saved_photo_names.append(f"{listing_id}/{save_name}")

        if photo.get("is_primary") or primary_photo_name is None:
            primary_photo_name = f"{listing_id}/{save_name}"

        # Evaluate photo quality using vision module
        try:
            rating = assess_photo_quality(content)
            photo_quality_scores.append(rating.get("quality_score", 70.0))
        except Exception:
            photo_quality_scores.append(75.0)

    avg_quality = float(np.mean(photo_quality_scores)) if photo_quality_scores else 75.0
    stars = max(1, min(5, int(round(avg_quality / 20.0))))

    record = {
        **metadata,
        "listing_id": listing_id,
        "photo_files": saved_photo_names,
        "primary_photo": primary_photo_name,
        "photo_quality_score": round(avg_quality, 1),
        "photo_stars": stars,
        "status": metadata.get("status", "Active"),
        "created_at": now_utc.isoformat(timespec="seconds"),
        "views_count": 0,
    }

    # Append to listings.json
    all_listings = get_all_broker_listings()
    all_listings.insert(0, record)
    LISTINGS_FILE.write_text(json.dumps(all_listings, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def delete_broker_listing(listing_id: str) -> bool:
    """Delete a broker listing and its uploaded photos from storage."""
    all_listings = get_all_broker_listings()
    filtered = [item for item in all_listings if item.get("listing_id") != listing_id]
    if len(filtered) == len(all_listings):
        return False

    LISTINGS_FILE.write_text(json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")

    # Remove photos directory
    listing_photos_dir = PHOTOS_DIR / listing_id
    if listing_photos_dir.exists():
        shutil.rmtree(listing_photos_dir, ignore_errors=True)
    return True


def update_broker_listing_status(listing_id: str, new_status: str) -> bool:
    """Update listing status: Active, Under Offer, Sold, Rented."""
    all_listings = get_all_broker_listings()
    updated = False
    for item in all_listings:
        if item.get("listing_id") == listing_id:
            item["status"] = new_status
            updated = True
            break
    if updated:
        LISTINGS_FILE.write_text(json.dumps(all_listings, indent=2, ensure_ascii=False), encoding="utf-8")
    return updated


def get_nearby_broker_listings(
    latitude: float,
    longitude: float,
    radius_km: float = 10.0,
    status: str = "Active",
) -> list[dict]:
    """Find active broker listings within a specified distance from target GPS coordinates."""
    listings = get_all_broker_listings(status=status)
    results = []
    target_point = (latitude, longitude)

    for item in listings:
        item_lat = item.get("latitude")
        item_lon = item.get("longitude")
        if item_lat is None or item_lon is None:
            continue
        try:
            dist = geodesic(target_point, (float(item_lat), float(item_lon))).kilometers
            if dist <= radius_km:
                item_copy = dict(item)
                item_copy["distance_km"] = round(dist, 2)
                results.append(item_copy)
        except (ValueError, TypeError):
            continue

    return sorted(results, key=lambda x: x["distance_km"])


def generate_smart_listing_copy(details: dict[str, Any]) -> dict[str, Any]:
    """
    Generate professional, high-converting listing headline, key highlights, and description.
    Uses OpenAI if OPENAI_API_KEY is configured, with a smart built-in template engine fallback.
    """
    # 1. Try external OpenAI generator if configured
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from ai_features import generate_property_listing
            return generate_property_listing(details, api_key=openai_key)
        except Exception:
            pass

    # 2. Intelligent Rule-Based Copywriting Engine
    title = details.get("title") or "Premium Residential Property"
    prop_type = details.get("property_type", "Apartment")
    bedrooms = details.get("bedrooms", 3)
    bathrooms = details.get("bathrooms", 2)
    area = details.get("area_sqft", 1500)
    furnishing = details.get("furnishing", "Semi-Furnished")
    locality = details.get("address") or "prime neighborhood"
    city = details.get("city") or "the city"
    amenities = details.get("amenities", [])
    floor = details.get("floor", 1)
    price_inr = details.get("price_inr", 0)

    # Headline
    bhk_text = f"{bedrooms} BHK" if bedrooms else "Spacious"
    headline = f"Exclusive {bhk_text} {prop_type} at {locality}, {city}"

    # Key highlights
    highlights = [
        f"Generous {area:,.0f} sq.ft layout with {bathrooms} modern bathrooms",
        f"{furnishing} finish with thoughtful architectural ventilation",
        f"Situated on floor {floor} offering excellent natural lighting",
    ]
    if amenities:
        amenity_summary = ", ".join(amenities[:3])
        highlights.append(f"Lifestyle amenities including {amenity_summary}")
    else:
        highlights.append("Secured community with dedicated parking and security")

    highlights.append(f"Prime connectivity in {locality}, {city}")

    # Description
    desc_lines = [
        f"Presenting a meticulously planned {bhk_text} {prop_type.lower()} positioned in {locality}, {city}.",
        f"Spanning an impressive super built-up area of {area:,.0f} square feet, this {furnishing.lower()} home provides the optimal balance of contemporary aesthetics and everyday functionality.",
        f"The property includes {bedrooms} expansive bedrooms, {bathrooms} well-appointed bathrooms, and bright open living spaces crafted for comfort.",
    ]
    if amenities:
        desc_lines.append(f"Residents enjoy premium access to {', '.join(amenities)}.")

    desc_lines.append("Strategically located with direct access to reputed educational institutions, premier shopping destinations, healthcare centers, and key transit arteries.")

    return {
        "headline": headline,
        "key_features": highlights,
        "description": " ".join(desc_lines),
    }


def reverse_geocode_coords(latitude: float, longitude: float) -> dict[str, str]:
    """Reverse geocode GPS coordinates to locality, city, state, and address string."""
    try:
        geolocator = Nominatim(user_agent="ai_estate_broker_portal", timeout=8)
        loc = geolocator.reverse((latitude, longitude), language="en")
        if not loc:
            return {"address": "", "city": "", "state": "", "pincode": ""}

        raw_addr = loc.raw.get("address", {})
        suburb = raw_addr.get("suburb") or raw_addr.get("neighbourhood") or raw_addr.get("residential") or ""
        city = raw_addr.get("city") or raw_addr.get("town") or raw_addr.get("county") or ""
        state = raw_addr.get("state") or ""
        pincode = raw_addr.get("postcode") or ""
        full_address = loc.address or ""

        return {
            "address": suburb or full_address.split(",")[0],
            "city": city,
            "state": state,
            "pincode": pincode,
            "full_address": full_address,
        }
    except Exception:
        return {"address": "", "city": "", "state": "", "pincode": "", "full_address": ""}
