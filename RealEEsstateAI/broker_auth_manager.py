"""
Broker Authentication and Credential Management Module for RealEEsstateAI.
Provides secure SQLite-backed broker registration, bcrypt password hashing with
legacy PBKDF2 verification,
credential verification against RERA/Username/Email, and a dedicated Broker Login UI.
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import html
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
import streamlit as st

from auth_manager import DB_PATH, hash_password, verify_password

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")


def get_db_connection():
    """Create a thread-safe connection to the authentication database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_broker_db():
    """Initialize the brokers table if it does not exist and seed the default broker account."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS brokers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                broker_id TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                agency_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                rera_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                status TEXT DEFAULT 'Active'
            );
            """
        )
        conn.commit()

        # Check if default demo broker account exists
        cursor = conn.execute("SELECT id FROM brokers WHERE username = 'vikram_broker'")
        if cursor.fetchone() is None:
            demo_pwd = "BrokerPassword123!"
            pwd_hash, salt = hash_password(demo_pwd)
            conn.execute(
                """
                INSERT INTO brokers (
                    broker_id, username, email, password_hash, salt,
                    full_name, agency_name, phone, rera_id, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "BRK-SEED-001",
                    "vikram_broker",
                    "vikram@primeestates.in",
                    pwd_hash,
                    salt,
                    "Vikram Malhotra",
                    "Prime Realty Advisors",
                    "+91 98450 12345",
                    "PRM/KA/RERA/1251/310/AG/210423/001890",
                    datetime.now(timezone.utc).isoformat(),
                    "Active",
                ),
            )
            conn.commit()


# Initialize table on module import
init_broker_db()


def register_broker(
    username: str,
    email: str,
    password: str,
    full_name: str,
    agency_name: str,
    phone: str,
    rera_id: str,
) -> tuple[bool, str]:
    """
    Register a new verified broker partner account.
    Returns (success: bool, message: str).
    """
    username = username.strip().lower()
    email = email.strip().lower()
    full_name = full_name.strip()
    agency_name = agency_name.strip()
    phone = phone.strip()
    rera_id = rera_id.strip()

    if not USERNAME_REGEX.match(username):
        return (
            False,
            "Broker username must be 3-30 characters long and contain only letters, numbers, and underscores.",
        )

    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid work email address."

    if len(password) < 6:
        return False, "Broker password must be at least 6 characters long."

    if not full_name:
        return False, "Please provide your full legal name."

    if not phone or len(phone) < 10:
        return False, "Please provide a valid 10-digit phone or WhatsApp number."

    if not rera_id:
        return False, "Official RERA registration ID is mandatory for broker accounts."

    pwd_hash, salt = hash_password(password)
    broker_id = f"BRK-ID-{uuid4().hex[:6].upper()}"

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO brokers (
                    broker_id, username, email, password_hash, salt,
                    full_name, agency_name, phone, rera_id, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    broker_id,
                    username,
                    email,
                    pwd_hash,
                    salt,
                    full_name,
                    agency_name or "Independent Broker",
                    phone,
                    rera_id,
                    datetime.now(timezone.utc).isoformat(),
                    "Active",
                ),
            )
            conn.commit()
        return True, "Broker partner account registered successfully! You can now log in."
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return False, f"Broker username '{username}' is already in use."
        elif "email" in err_msg:
            return False, f"Email address '{email}' is already registered."
        return False, "A broker with these credentials already exists."
    except Exception as e:
        return False, f"Database error during broker registration: {e}"


def authenticate_broker(
    identifier: str,
    password: str,
) -> tuple[bool, dict | str]:
    """
    Authenticate a broker using their username, work email, or official RERA ID.
    Returns (success: bool, broker_data_or_error: dict | str).
    """
    id_clean = identifier.strip().lower()
    if not id_clean or not password:
        return False, "Please provide both identifier and password."

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, broker_id, username, email, password_hash, salt,
                       full_name, agency_name, phone, rera_id, status
                FROM brokers
                WHERE LOWER(username) = ? OR LOWER(email) = ? OR LOWER(rera_id) = ?
                """,
                (id_clean, id_clean, id_clean),
            )
            row = cursor.fetchone()

            if row is None:
                return False, "Invalid broker credentials. Check username, email, or RERA ID."

            if not verify_password(password, row["salt"], row["password_hash"]):
                return False, "Incorrect broker password."

            if row["status"] != "Active":
                return False, "Your broker account is currently suspended. Please contact support."

            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute("UPDATE brokers SET last_login = ? WHERE id = ?", (now_iso, row["id"]))
            conn.commit()

            broker_data = {
                "id": row["id"],
                "broker_id": row["broker_id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "agency_name": row["agency_name"],
                "phone": row["phone"],
                "rera_id": row["rera_id"],
                "last_login": now_iso,
            }
            return True, broker_data
    except Exception as e:
        return False, f"Authentication error: {e}"


def list_registered_brokers() -> list[dict]:
    """Return broker profile details without authentication secrets for admin review."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT broker_id, username, email, full_name, agency_name,
                   phone, rera_id, created_at, last_login, status
            FROM brokers
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def is_broker_authenticated() -> bool:
    """Check if the current session has an authenticated broker."""
    return st.session_state.get("broker_authenticated", False) is True


def get_current_broker() -> dict | None:
    """Get the currently logged-in broker dictionary, or None."""
    if is_broker_authenticated():
        return st.session_state.get("broker_user")
    return None


def login_broker(broker_data: dict):
    """Set active broker session state."""
    st.session_state["broker_authenticated"] = True
    st.session_state["broker_user"] = broker_data


def logout_broker():
    """Clear broker authentication and rerun."""
    st.session_state["broker_authenticated"] = False
    st.session_state.pop("broker_user", None)
    st.rerun()


def render_broker_login_card():
    """Renders a sleek, dedicated Broker Authentication Gate card."""
    st.markdown(
        """
        <div style="max-width: 650px; margin: 30px auto 20px auto; padding: 30px 34px; background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95)); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 20px; box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5);">
            <div style="text-align: center; margin-bottom: 22px;">
                <div style="display: inline-block; background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; border-radius: 50%; padding: 14px 18px; margin-bottom: 10px;">
                    <span style="font-size: 2rem;">🏢</span>
                </div>
                <h2 style="margin: 0; color: #f8fafc; font-size: 1.65rem; font-weight: 700;">Verified Broker & Agent Portal</h2>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.92rem;">
                    Sign in with your RERA-registered broker credentials to publish listings, manage property inventory, and access buyer leads.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_col1, auth_col2, auth_col3 = st.columns([1, 4, 1])
    with auth_col2:
        auth_tab_login, auth_tab_register = st.tabs(["🔑 Broker Login", "📝 Register New Broker Partner"])

        with auth_tab_login:
            st.markdown("##### Sign In with Registered Broker Credentials")
            login_id = st.text_input(
                "Username, Work Email, or RERA ID",
                placeholder="e.g. vikram_broker or PRM/KA/RERA/...",
                key="broker_auth_identifier",
            )
            login_pwd = st.text_input(
                "Broker Password",
                type="password",
                placeholder="Enter your broker password",
                key="broker_auth_password",
            )

            col_submit, col_demo = st.columns([1, 1])
            with col_submit:
                if st.button("🔓 Sign In as Broker", type="primary", use_container_width=True, key="broker_login_submit_btn"):
                    if not login_id or not login_pwd:
                        st.error("Please enter both identifier and password.")
                    else:
                        success, res = authenticate_broker(login_id, login_pwd)
                        if success:
                            login_broker(res)
                            st.success(f"Welcome back, {res['full_name']}!")
                            st.rerun()
                        else:
                            st.error(res)

            with col_demo:
                if st.button("⚡ Quick Demo Broker Login", use_container_width=True, key="broker_demo_login_btn", help="Login instantly as Vikram Malhotra (Prime Realty)"):
                    success, res = authenticate_broker("vikram_broker", "BrokerPassword123!")
                    if success:
                        login_broker(res)
                        st.success("Signed in with demo broker account!")
                        st.rerun()
                    else:
                        st.error(res)

            st.caption("Default Demo Credentials: Username `vikram_broker` · Password `BrokerPassword123!`")

        with auth_tab_register:
            st.markdown("##### Apply for Certified Broker Partner Access")
            st.caption("All listings published via this portal are tagged with verified broker credentials and official RERA ID.")

            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                reg_name = st.text_input("Full Legal Name*", placeholder="e.g. Rajesh Sharma", key="broker_reg_name")
                reg_agency = st.text_input("Agency / Brokerage Firm*", placeholder="e.g. Apex Realty Group", key="broker_reg_agency")
                reg_phone = st.text_input("Phone / WhatsApp*", placeholder="e.g. +91 98765 43210", key="broker_reg_phone")
            with reg_c2:
                reg_rera = st.text_input("RERA Registration Number*", placeholder="e.g. PRM/KA/RERA/1251/...", key="broker_reg_rera")
                reg_email = st.text_input("Work Email Address*", placeholder="e.g. rajesh@apexrealty.com", key="broker_reg_email")
                reg_uname = st.text_input("Choose Username*", placeholder="e.g. rajesh_broker", key="broker_reg_username")

            reg_pwd = st.text_input("Create Password (min. 6 chars)*", type="password", key="broker_reg_pwd")

            if st.button("📝 Register as Broker Partner", type="primary", use_container_width=True, key="broker_reg_submit_btn"):
                ok, msg = register_broker(
                    username=reg_uname,
                    email=reg_email,
                    password=reg_pwd,
                    full_name=reg_name,
                    agency_name=reg_agency,
                    phone=reg_phone,
                    rera_id=reg_rera,
                )
                if ok:
                    st.success(msg)
                    st.info("You can now switch to the 'Broker Login' tab to sign in.")
                else:
                    st.error(msg)
