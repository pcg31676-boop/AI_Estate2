"""
Verification Suite for Broker Portal, Leaflet Geo-tagging, and Live Photo Capture.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from io import BytesIO
import numpy as np
from PIL import Image

import broker_manager
from vision_module import assess_photo_quality

print("--- [1/4] Testing Broker Storage & Seed Data ---")
listings = broker_manager.get_all_broker_listings()
assert len(listings) >= 3, f"Expected at least 3 seed listings, got {len(listings)}"
print(f"✔ Initialized broker repository: {len(listings)} listings active")

print("\n--- [2/4] Testing Live Photo Capture & AI Quality Rating ---")
# Create synthetic sample image
img = Image.new("RGB", (640, 480), color=(180, 160, 140))
buf = BytesIO()
img.save(buf, format="JPEG")
test_photo_bytes = buf.getvalue()

quality_res = assess_photo_quality(test_photo_bytes)
assert "quality_score" in quality_res
assert "stars" in quality_res
print(f"✔ Photo quality evaluated: {quality_res['quality_score']:.1f}/100 ({quality_res['stars']} stars)")

print("\n--- [3/4] Testing Broker Listing Creation with Leaflet Geo-tagging ---")
test_metadata = {
    "title": "Automated Test Penthouse",
    "property_type": "Penthouse",
    "listing_type": "For Sale",
    "price_inr": 25000000,
    "bedrooms": 4,
    "bathrooms": 4,
    "balconies": 3,
    "area_sqft": 3200.0,
    "furnishing": "Furnished",
    "floor": 18,
    "total_floors": 20,
    "amenities": ["Covered Parking", "Swimming Pool", "Gym", "Private Terrace"],
    "latitude": 12.9352,
    "longitude": 77.6245,
    "address": "80 Feet Road, Koramangala",
    "city": "Bengaluru",
    "state": "Karnataka",
    "pincode": "560034",
    "broker_name": "Test Broker",
    "broker_phone": "+91 99999 11111",
    "broker_agency": "Apex Real Estate",
    "rera_id": "TEST/RERA/001",
    "headline": "Stunning 4 BHK Penthouse",
    "description": "Exclusive top-floor living with private terrace and panoramic vistas.",
}

photos_payload = [
    {"filename": "test_living.jpg", "bytes": test_photo_bytes, "is_primary": True}
]

created = broker_manager.save_broker_listing(test_metadata, photos_payload)
assert created["listing_id"].startswith("BRK-"), "Listing ID must start with BRK-"
assert len(created["photo_files"]) == 1, "Must have 1 saved photo file"
print(f"✔ Listing created: {created['listing_id']} at ({created['latitude']}, {created['longitude']})")

# Test nearby query
nearby = broker_manager.get_nearby_broker_listings(12.9350, 77.6240, radius_km=5.0)
assert any(item["listing_id"] == created["listing_id"] for item in nearby), "Newly created listing must be found in nearby query"
print(f"✔ Leaflet proximity search verified: found {len(nearby)} listings within 5 km")

# Test AI Copywriting generator
copy_result = broker_manager.generate_smart_listing_copy(test_metadata)
assert "headline" in copy_result
assert "key_features" in copy_result
assert "description" in copy_result
print(f"✔ AI Marketing Copy generated: '{copy_result['headline']}'")

print("\n--- [4/5] Testing Status Update & Deletion ---")
up_ok = broker_manager.update_broker_listing_status(created["listing_id"], "Sold")
assert up_ok, "Status update must succeed"
updated_item = broker_manager.get_broker_listing(created["listing_id"])
assert updated_item["status"] == "Sold", "Status must now be 'Sold'"
print("✔ Status updated to 'Sold'")

del_ok = broker_manager.delete_broker_listing(created["listing_id"])
assert del_ok, "Delete must succeed"
assert broker_manager.get_broker_listing(created["listing_id"]) is None, "Listing must no longer exist"
print("✔ Cleaned up test listing successfully")

print("\n--- [5/5] Testing Broker Authentication & Credentials Interface ---")
import broker_auth_manager

# 1. Test Seed account authentication via Username, Email, and RERA ID
auth_by_user, broker_obj = broker_auth_manager.authenticate_broker("vikram_broker", "BrokerPassword123!")
assert auth_by_user, f"Expected successful login for vikram_broker, got {broker_obj}"
assert broker_obj["full_name"] == "Vikram Malhotra"
print("✔ Authenticated demo broker by Username ('vikram_broker')")

auth_by_email, broker_obj2 = broker_auth_manager.authenticate_broker("vikram@primeestates.in", "BrokerPassword123!")
assert auth_by_email, "Authentication by work email must succeed"
assert broker_obj2["username"] == "vikram_broker"
print("✔ Authenticated demo broker by Work Email ('vikram@primeestates.in')")

auth_by_rera, broker_obj3 = broker_auth_manager.authenticate_broker("PRM/KA/RERA/1251/310/AG/210423/001890", "BrokerPassword123!")
assert auth_by_rera, "Authentication by RERA ID must succeed"
assert broker_obj3["username"] == "vikram_broker"
print("✔ Authenticated demo broker by RERA Registration ID")

# 2. Test Invalid Password Rejection
bad_auth, bad_err = broker_auth_manager.authenticate_broker("vikram_broker", "WrongPassword!")
assert not bad_auth, "Bad password must fail"
assert "Incorrect" in bad_err or "Invalid" in bad_err
print(f"✔ Rejected incorrect credentials properly: '{bad_err}'")

# 3. Test Registration of New Broker
import uuid
unique_user = f"broker_{uuid.uuid4().hex[:6]}"
unique_email = f"{unique_user}@realtynetwork.com"
reg_ok, reg_msg = broker_auth_manager.register_broker(
    username=unique_user,
    email=unique_email,
    password="NewSecureBrokerPass123!",
    full_name="Ananya Sharma",
    agency_name="Skyline Realty Partners",
    phone="+91 91234 56789",
    rera_id=f"PRM/KA/RERA/{uuid.uuid4().hex[:4].upper()}",
)
assert reg_ok, f"Registration must succeed, got {reg_msg}"
print(f"✔ Registered new verified broker account: '{unique_user}'")

# Test duplicate prevention
dup_ok, dup_msg = broker_auth_manager.register_broker(
    username=unique_user,
    email=f"other_{unique_email}",
    password="Pass123456!",
    full_name="Duplicate",
    agency_name="Agency",
    phone="+91 99999 88888",
    rera_id="RERA123",
)
assert not dup_ok, "Duplicate username must be rejected"
assert "already in use" in dup_msg.lower() or "already exists" in dup_msg.lower()
print(f"✔ Duplicate broker registration prevented: '{dup_msg}'")

print("\n🎉 ALL BROKER PORTAL & AUTHENTICATION UNIT TESTS PASSED WITH 100% SUCCESS!")
