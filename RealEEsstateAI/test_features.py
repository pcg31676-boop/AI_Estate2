"""
Automated Verification Suite for Authentication and Photo Analysis features.
Executes non-regression tests against existing models, dataset loaders,
and verifies security of the auth subsystem and vision analysis pipeline.
"""

import os
import sys
import sqlite3
import hashlib
from unittest.mock import patch
from io import BytesIO
import numpy as np
from PIL import Image

# 1. Test Authentication Subsystem
print("--- [1/3] Testing Authentication Subsystem ---")
import auth_manager

# Test password hashing
pwd = "SuperSecretPassword123!"
hash1, salt1 = auth_manager.hash_password(pwd)
assert hash1.startswith(("$2a$", "$2b$", "$2y$")), "New hashes must use bcrypt"
assert len(salt1) == 29, "Stored bcrypt salt must use the bcrypt salt format"
assert auth_manager.verify_password(pwd, salt1, hash1), "Valid password must verify"
assert not auth_manager.verify_password("WrongPassword!", salt1, hash1), "Wrong password must fail"

legacy_salt = "01" * 32
legacy_hash = hashlib.pbkdf2_hmac(
    "sha256", pwd.encode("utf-8"), bytes.fromhex(legacy_salt),
    auth_manager.PBKDF2_ITERATIONS,
).hex()
assert auth_manager.verify_password(
    pwd, legacy_salt, legacy_hash
), "Existing PBKDF2 account hashes must remain valid"

# Test Registration & Validation
test_username = "test_user_ai"
test_email = "test_user_ai@example.com"

# Clean up if already exists in test DB
with auth_manager.get_db_connection() as conn:
    conn.execute("DELETE FROM users WHERE username = ? OR email = ?", (test_username, test_email))
    conn.commit()

# Valid registration
ok, msg = auth_manager.register_user(test_username, test_email, pwd)
assert ok, f"Registration failed: {msg}"
print(f"✔ Registration successful: {msg}")
with auth_manager.get_db_connection() as conn:
    new_user = conn.execute(
        "SELECT role, password_hash FROM users WHERE username = ?",
        (test_username,),
    ).fetchone()
assert new_user["role"] == "user", "Public registration must not create admin accounts"
assert new_user["password_hash"].startswith("$2"), "New account passwords must be bcrypt-hashed"
with patch.dict(
    os.environ,
    {
        "AI_ESTATE_ADMIN_USERNAME": test_username,
        "AI_ESTATE_ADMIN_PASSWORD": "SecureAdminPassword123!",
    },
):
    reserved_ok, _ = auth_manager.register_user(
        test_username, "reserved_admin@example.com", pwd
    )
    assert not reserved_ok, "Public registration must not claim configured admin username"

# Duplicate username rejection
dup_ok, dup_msg = auth_manager.register_user(test_username, "different@example.com", pwd)
assert not dup_ok, "Duplicate username must be rejected"
print(f"✔ Duplicate username correctly rejected: {dup_msg}")

# Duplicate email rejection
dup_email_ok, dup_email_msg = auth_manager.register_user("different_user", test_email, pwd)
assert not dup_email_ok, "Duplicate email must be rejected"
print(f"✔ Duplicate email correctly rejected: {dup_email_msg}")

# Invalid email validation
inv_email_ok, inv_email_msg = auth_manager.register_user("validuser", "invalid-email-format", pwd)
assert not inv_email_ok, "Invalid email format must be rejected"
print(f"✔ Invalid email format correctly rejected: {inv_email_msg}")

# Short password rejection
short_pwd_ok, short_pwd_msg = auth_manager.register_user("validuser2", "valid2@example.com", "123")
assert not short_pwd_ok, "Short password (<6 chars) must be rejected"
print(f"✔ Short password correctly rejected: {short_pwd_msg}")

# Authentication tests
auth_ok, user_data = auth_manager.authenticate_user(test_username, pwd)
assert auth_ok, f"Valid authentication by username failed: {user_data}"
assert user_data["username"] == test_username
assert user_data["role"] == "user", "Public registration must create a regular user account"
print(f"✔ Authenticated by username: {user_data['username']}")

admin_login_ok, _ = auth_manager.authenticate_user(
    test_username, pwd, required_role="admin"
)
assert not admin_login_ok, "A regular user account must not authenticate as an admin"
print("✔ Regular user credentials rejected by the admin role gate")

admin_conn = sqlite3.connect(":memory:", check_same_thread=False)
admin_conn.row_factory = sqlite3.Row
admin_environment = {
    "AI_ESTATE_ADMIN_USERNAME": "test_admin",
    "AI_ESTATE_ADMIN_PASSWORD": "SecureAdminPassword123!",
    "AI_ESTATE_ADMIN_EMAIL": "test_admin@example.com",
}
expected_configuration_error = (
    "Admin access is not configured. Set AI_ESTATE_ADMIN_USERNAME and "
    "AI_ESTATE_ADMIN_PASSWORD in the server environment."
)
with patch.object(
    auth_manager, "_admin_setting", side_effect=lambda name, default="": default
):
    missing_admin_ok, missing_admin_result = auth_manager.authenticate_admin(
        "test_admin", "SecureAdminPassword123!"
    )
    assert not missing_admin_ok
    assert missing_admin_result == expected_configuration_error
with patch.dict(
    os.environ,
    {
        "AI_ESTATE_ADMIN_USERNAME": "test_admin",
        "AI_ESTATE_ADMIN_PASSWORD": "short",
    },
):
    assert (
        auth_manager.admin_configuration_error() == expected_configuration_error
    ), "Missing password-strength requirements must show the setup error"
try:
    with patch.object(
        auth_manager, "get_db_connection", return_value=admin_conn
    ), patch.dict(os.environ, admin_environment):
        auth_manager.init_db()
        admin_ok, admin_data = auth_manager.authenticate_admin(
            "test_admin", "SecureAdminPassword123!"
        )
        assert admin_ok, f"Configured admin login failed: {admin_data}"
        assert admin_data["role"] == "admin"
        assert not auth_manager.authenticate_admin(
            "wrong_admin", "SecureAdminPassword123!"
        )[0], "Admin username comparison must reject mismatches"
        assert not auth_manager.authenticate_admin(
            "test_admin", "WrongAdminPassword!"
        )[0], "Admin login must reject an incorrect password"
        assert not auth_manager.authenticate_user(
            "test_admin", "SecureAdminPassword123!"
        )[0], "Admin accounts must not authenticate through the user login"
finally:
    admin_conn.close()
print("✔ Environment-provisioned admin login is isolated from user login")

auth_email_ok, user_data2 = auth_manager.authenticate_user(test_email, pwd)
assert auth_email_ok, f"Valid authentication by email failed: {user_data2}"
assert user_data2["email"] == test_email
print(f"✔ Authenticated by email: {user_data2['email']}")

auth_bad_pwd_ok, err_msg = auth_manager.authenticate_user(test_username, "WrongPassword")
assert not auth_bad_pwd_ok, "Authentication with wrong password must fail"
print(f"✔ Invalid password authentication correctly rejected: {err_msg}")

auth_nonexistent_ok, err_msg2 = auth_manager.authenticate_user("nonexistent_user_999", pwd)
assert not auth_nonexistent_ok, "Nonexistent user authentication must fail"
print(f"✔ Nonexistent user authentication correctly rejected: {err_msg2}")

print("✔ Auth subsystem passed all checks!\n")


# 2. Test Photo Analysis Subsystem
print("--- [2/3] Testing Photo Analysis Subsystem ---")
import photo_analyzer


class DummyUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


# Test corrupted file
corrupt_upload = DummyUpload("broken.jpg", b"not-a-valid-image-bytes")
val_ok, val_err, _ = photo_analyzer.validate_image_file(corrupt_upload)
assert not val_ok, "Corrupted image file must fail validation"
print(f"✔ Corrupted image correctly rejected: {val_err}")

# Test image that is too small (<64x64)
tiny_img = Image.new("RGB", (32, 32), color=(255, 0, 0))
tiny_buf = BytesIO()
tiny_img.save(tiny_buf, format="JPEG")
tiny_upload = DummyUpload("tiny.jpg", tiny_buf.getvalue())
tiny_ok, tiny_err, _ = photo_analyzer.validate_image_file(tiny_upload)
assert not tiny_ok, "Tiny image (<64x64) must fail validation"
print(f"✔ Tiny image correctly rejected: {tiny_err}")

# Test real sample photo from workspace
sample_photo_path = os.path.join("images", "1000102", "bedroom.jpg")
assert os.path.isfile(sample_photo_path), f"Sample photo not found: {sample_photo_path}"

with open(sample_photo_path, "rb") as f:
    sample_bytes = f.read()

real_upload = DummyUpload("bedroom.jpg", sample_bytes)
real_ok, real_err, meta = photo_analyzer.validate_image_file(real_upload)
assert real_ok, f"Real photo validation failed: {real_err}"
print(f"✔ Real photo validated: {meta['width']}x{meta['height']} ({meta['format']}), {meta['size_kb']:.1f} KB")

# Test deep neural analysis pipeline on real bedroom photo
analysis = photo_analyzer.analyze_single_photo(sample_bytes)
assert "room_type" in analysis
assert "top_predictions" in analysis
assert len(analysis["top_predictions"]) == 5
assert "quality_score" in analysis
assert 0 <= analysis["quality_score"] <= 100
assert 1 <= analysis["stars"] <= 5
print(f"✔ Neural analysis completed:")
print(f"   Room Type Classified: {analysis['room_type']}")
print(f"   Primary Prediction: {analysis['primary_label']} ({analysis['primary_confidence_pct']}%)")
print(f"   Quality Score: {analysis['quality_score']}/100, Finish: {analysis['finish_score']}/5.0 ({analysis['stars']} stars)")
print(f"   Top 3 predictions: {[p['label'] + ': ' + str(p['confidence_pct']) + '%' for p in analysis['top_predictions'][:3]]}")

print("✔ Photo Analysis subsystem passed all checks!\n")


# 3. Test Baseline Application & Model Integrity
print("--- [3/3] Testing Existing Models & Dataset Integrity ---")
import pandas as pd
from price_model import MODEL_PATH, load_or_train_model
from train_land_model import LAND_MODEL_PATH, load_land_model

# Check properties.csv
prop_df = pd.read_csv("properties.csv")
assert not prop_df.empty, "properties.csv must not be empty"
print(f"✔ properties.csv loaded: {len(prop_df)} rows")

# Check land dataset
land_df = pd.read_csv("sample_land_schema.csv")
assert not land_df.empty, "sample_land_schema.csv must not be empty"
print(f"✔ sample_land_schema.csv loaded: {len(land_df)} rows")

# Check price model
bundle = load_or_train_model(prop_df, "images", MODEL_PATH)
assert "model" in bundle and "feature_columns" in bundle
print(f"✔ Price model verified: {len(bundle['feature_columns'])} features")

# Check land model
land_bundle = load_land_model(LAND_MODEL_PATH, allow_synthetic=True)
assert "model" in land_bundle
print(f"✔ Land valuation model verified")

print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! ZERO REGRESSIONS DETECTED.")
