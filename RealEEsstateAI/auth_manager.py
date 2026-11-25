"""
Authentication and User Management Module for Real Estate Valuation AI.
Provides secure SQLite-backed accounts, bcrypt password hashing with PBKDF2
verification for existing accounts, session management, and Streamlit UI.
"""

import os
import re
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime
import bcrypt
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth.db")
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
PBKDF2_ITERATIONS = 200_000


def get_db_connection():
    """Create a thread-safe connection to the authentication database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the users table if it does not exist."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            """
        )
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "role" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
            )
        conn.commit()


# Initialize table on import
init_db()


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """
    Hash new passwords with bcrypt. When a salt is supplied, retain the legacy
    PBKDF2 behavior for callers that need to reproduce an existing hash.
    """
    if salt is not None:
        salt_bytes = bytes.fromhex(salt)
        hash_bytes = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS
        )
        return hash_bytes.hex(), salt

    bcrypt_input = hashlib.sha256(
        b"AI Estate password v1\x00" + password.encode("utf-8")
    ).hexdigest().encode("ascii")
    password_hash = bcrypt.hashpw(
        bcrypt_input, bcrypt.gensalt(rounds=12)
    ).decode("ascii")
    return password_hash, password_hash[:29]


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify bcrypt hashes and legacy PBKDF2 hashes."""
    if expected_hash.startswith(("$2a$", "$2b$", "$2y$")):
        bcrypt_input = hashlib.sha256(
            b"AI Estate password v1\x00" + password.encode("utf-8")
        ).hexdigest().encode("ascii")
        return bcrypt.checkpw(bcrypt_input, expected_hash.encode("ascii"))

    salt_bytes = bytes.fromhex(salt)
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS
    )
    return secrets.compare_digest(hash_bytes.hex(), expected_hash)


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user after validation.
    Returns (success: bool, message: str).
    """
    username = username.strip().lower()
    email = email.strip().lower()
    reserved_admin_username = _admin_setting("AI_ESTATE_ADMIN_USERNAME").strip().lower()

    if not USERNAME_REGEX.match(username):
        return (
            False,
            "Username must be 3-30 characters long and contain only letters, numbers, and underscores.",
        )
    if reserved_admin_username and hmac.compare_digest(
        username.encode("utf-8"), reserved_admin_username.encode("utf-8")
    ):
        return False, "That username is reserved for the administrator account."

    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    pwd_hash, salt = hash_password(password)

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, 'user', ?)
                """,
                (username, email, pwd_hash, salt, datetime.now().isoformat()),
            )
            conn.commit()
        return True, "Account created successfully! You can now log in."
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return False, f"Username '{username}' is already taken."
        elif "email" in err_msg:
            return False, f"Email '{email}' is already registered."
        return False, "Username or email is already in use."
    except Exception as e:
        return False, f"Database error during registration: {e}"


def authenticate_user(
    username_or_email: str,
    password: str,
    required_role: str = "user",
) -> tuple[bool, dict | str]:
    """
    Authenticate a user by username or email.
    Returns (success: bool, user_data_or_error: dict | str).
    """
    identifier = username_or_email.strip().lower()
    if not identifier or not password:
        return False, "Please provide both credentials."

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, username, email, password_hash, salt, role
                FROM users
                WHERE (username = ? OR email = ?) AND role = ?
                """,
                (identifier, identifier, required_role),
            )
            row = cursor.fetchone()

            if row is None:
                return False, "Invalid username/email or password."

            if not verify_password(password, row["salt"], row["password_hash"]):
                return False, "Invalid username/email or password."

            # Update last login timestamp
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (now, row["id"]),
            )
            conn.commit()

            user_data = {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "role": row["role"],
                "last_login": now,
            }
            return True, user_data
    except Exception as e:
        return False, f"Authentication error: {e}"


def _admin_setting(name: str, default: str = "") -> str:
    """Read an admin setting from the environment, then Streamlit secrets."""
    environment_value = os.environ.get(name)
    if environment_value:
        return environment_value

    try:
        secret_value = st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default
    return str(secret_value) if secret_value is not None else default


def admin_configuration_error() -> str | None:
    """Return setup guidance when the environment- or secrets-provisioned admin is unavailable."""
    username = _admin_setting("AI_ESTATE_ADMIN_USERNAME").strip().lower()
    password = _admin_setting("AI_ESTATE_ADMIN_PASSWORD")
    configuration_message = (
        "Admin access is not configured. Set AI_ESTATE_ADMIN_USERNAME and "
        "AI_ESTATE_ADMIN_PASSWORD in the server environment."
    )
    if not username or not password.strip() or len(password) < 12:
        return configuration_message
    if not USERNAME_REGEX.match(username):
        return "AI_ESTATE_ADMIN_USERNAME must be 3-30 letters, numbers, or underscores."
    admin_email = _admin_setting(
        "AI_ESTATE_ADMIN_EMAIL", f"{username}@admin.local"
    ).strip().lower()
    if not EMAIL_REGEX.match(admin_email):
        return "AI_ESTATE_ADMIN_EMAIL must be a valid email address."
    return None


def authenticate_admin(identifier: str, password: str) -> tuple[bool, dict | str]:
    """Authenticate the configured admin using constant-time credential comparisons."""
    config_error = admin_configuration_error()
    if config_error:
        return False, config_error

    admin_username = _admin_setting("AI_ESTATE_ADMIN_USERNAME").strip().lower()
    admin_password = _admin_setting("AI_ESTATE_ADMIN_PASSWORD")
    admin_email = _admin_setting(
        "AI_ESTATE_ADMIN_EMAIL", f"{admin_username}@admin.local"
    ).strip().lower()

    username_matches = hmac.compare_digest(
        identifier.strip().lower().encode("utf-8"),
        admin_username.encode("utf-8"),
    )
    password_matches = hmac.compare_digest(
        password.encode("utf-8"),
        admin_password.encode("utf-8"),
    )
    if not (username_matches and password_matches):
        return False, "Invalid admin username or password"

    pwd_hash, salt = hash_password(admin_password)
    try:
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT id, email, role FROM users WHERE username = ?",
                (admin_username,),
            ).fetchone()
            email_owner = conn.execute(
                "SELECT username FROM users WHERE email = ?", (admin_email,)
            ).fetchone()
            if email_owner and email_owner["username"] != admin_username:
                return False, "The configured admin email is already assigned to another account."

            if existing:
                if existing["role"] != "admin":
                    return False, "The configured admin username is already assigned to a user account."
                conn.execute(
                    """
                    UPDATE users
                    SET email = ?, password_hash = ?, salt = ?, role = 'admin'
                    WHERE id = ?
                    """,
                    (admin_email, pwd_hash, salt, existing["id"]),
                )
                admin_id = existing["id"]
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, email, password_hash, salt, role, created_at)
                    VALUES (?, ?, ?, ?, 'admin', ?)
                    """,
                    (
                        admin_username,
                        admin_email,
                        pwd_hash,
                        salt,
                        datetime.now().isoformat(),
                    ),
                )
                admin_id = cursor.lastrowid

            now = datetime.now().isoformat()
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, admin_id))
            conn.commit()
            return True, {
                "id": admin_id,
                "username": admin_username,
                "email": admin_email,
                "role": "admin",
                "last_login": now,
            }
    except Exception as e:
        return False, f"Admin authentication error: {e}"


def list_registered_users() -> list[dict]:
    """Return public account details for the admin dashboard."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, username, email, role, created_at, last_login
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def is_authenticated() -> bool:
    """Check if the current session is authenticated."""
    return st.session_state.get("authenticated", False) is True


def get_current_user() -> dict | None:
    """Get the current authenticated user dictionary or None."""
    if is_authenticated():
        return st.session_state.get("auth_user")
    return None


def logout_user():
    """Clear authentication session and trigger rerun."""
    st.session_state["authenticated"] = False
    st.session_state.pop("auth_user", None)
    st.session_state.pop("role", None)
    st.rerun()


def render_auth_ui(background_image: str | None = None) -> bool:
    """
    Renders the modern Login & Registration card.
    Returns True if user is authenticated, False if login is required.
    """
    if is_authenticated():
        return True

    # Page styling for modern authentication portal
    st.markdown(
        """
        <style>
        .auth-container {
            max-width: 480px;
            margin: 40px auto 20px auto;
            padding: 32px 36px;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(12px);
            color: #F8FAFC;
        }
        .auth-header {
            text-align: center;
            margin-bottom: 24px;
        }
        .auth-logo {
            font-size: 2.8rem;
            margin-bottom: 8px;
        }
        .auth-title {
            font-size: 1.7rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            margin: 0;
        }
        .auth-subtitle {
            font-size: 0.92rem;
            color: #94A3B8;
            margin-top: 6px;
        }
        .auth-badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.2);
            color: #60A5FA;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 10px;
            border: 1px solid rgba(59, 130, 246, 0.4);
        }
        .st-key-admin_login_submit button {
            background: linear-gradient(135deg, #ef4444, #b91c1c) !important;
            border: 1px solid #f87171 !important;
            color: #fff !important;
        }
        .st-key-admin_login_submit button:hover {
            background: linear-gradient(135deg, #dc2626, #991b1b) !important;
            border-color: #fca5a5 !important;
        }
        @media (max-width: 600px) {
            .auth-container {
                margin: 20px auto 16px;
                padding: 24px 18px;
            }
            .auth-title {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown(
            """
            <div class="auth-container">
                <div class="auth-header">
                    <div class="auth-logo">🏠</div>
                    <h2 class="auth-title">AI Estate Valuation</h2>
                    <div class="auth-subtitle">Enterprise Property Valuation & PropTech Platform</div>
                    <span class="auth-badge">🔒 Secure Gateway</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_admin, tab_register = st.tabs(
            ["👤 User Sign In", "🛡️ Admin Sign In", "📝 Create Account"]
        )

        with tab_login:
            st.markdown("##### Welcome back! Sign in to continue")
            login_id = st.text_input(
                "Username or Email",
                placeholder="e.g. john or john@example.com",
                key="login_identifier",
            )
            login_pw = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )
            if st.button("Sign In to Dashboard", type="primary", use_container_width=True):
                if not login_id or not login_pw:
                    st.error("Please enter both username/email and password.")
                else:
                    with st.spinner("Verifying credentials..."):
                        success, result = authenticate_user(login_id, login_pw)
                        if success:
                            st.session_state["authenticated"] = True
                            st.session_state["auth_user"] = result
                            st.session_state["role"] = result["role"]
                            st.success(f"Welcome back, {result['username']}! Redirecting...")
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")

        with tab_admin:
            st.markdown("##### Secure administrator access")
            st.caption(
                "Admin accounts are provisioned by the server operator; public registration "
                "cannot create administrator accounts. Configure "
                "AI_ESTATE_ADMIN_USERNAME and a 12+ character "
                "AI_ESTATE_ADMIN_PASSWORD on the server."
            )
            admin_id = st.text_input(
                "Admin Username",
                placeholder="Configured server admin username",
                key="admin_login_identifier",
            )
            admin_password = st.text_input(
                "Admin Password",
                type="password",
                placeholder="Enter your admin password",
                key="admin_login_password",
            )
            if st.button(
                "Sign In as Admin",
                type="primary",
                use_container_width=True,
                key="admin_login_submit",
            ):
                config_error = admin_configuration_error()
                if config_error:
                    st.error(config_error)
                elif not admin_id or not admin_password:
                    st.error("Please enter both admin username and password.")
                else:
                    success, result = authenticate_admin(admin_id, admin_password)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["auth_user"] = result
                        st.session_state["role"] = "admin"
                        st.rerun()
                    else:
                        st.error(result)

        with tab_register:
            st.markdown("##### New to AI Estate? Register below")
            reg_username = st.text_input(
                "Choose Username",
                placeholder="3-30 alphanumeric characters",
                key="reg_username",
            )
            reg_email = st.text_input(
                "Email Address",
                placeholder="name@example.com",
                key="reg_email",
            )
            reg_pw = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 6 characters",
                key="reg_password",
            )
            reg_pw_confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter password",
                key="reg_password_confirm",
            )
            if st.button("Create Account", type="primary", use_container_width=True):
                if not reg_username or not reg_email or not reg_pw:
                    st.error("Please fill out all registration fields.")
                elif reg_pw != reg_pw_confirm:
                    st.error("Passwords do not match. Please re-enter carefully.")
                else:
                    with st.spinner("Creating your secure account..."):
                        success, message = register_user(
                            reg_username, reg_email, reg_pw
                        )
                        if success:
                            # Auto-login the newly registered user
                            auth_ok, auth_data = authenticate_user(
                                reg_username, reg_pw
                            )
                            if auth_ok:
                                st.session_state["authenticated"] = True
                                st.session_state["auth_user"] = auth_data
                                st.session_state["role"] = auth_data["role"]
                                st.success("Account created and verified! Redirecting to dashboard...")
                                st.rerun()
                            else:
                                st.success(message)
                        else:
                            st.error(f"❌ {message}")

    return False


def render_user_navbar():
    """Renders a sleek top bar with user profile info and logout button."""
    user = get_current_user()
    if not user:
        return

    st.markdown(
        """
        <style>
        .user-nav-box {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 18px;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            margin-bottom: 16px;
        }
        .user-nav-info {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.95rem;
            color: #E2E8F0;
        }
        .user-nav-badge {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_col1, nav_col2 = st.columns([5, 1])
    with nav_col1:
        role = str(user.get("role", "user")).title()
        st.markdown(
            f"""
            <div class="user-nav-box">
                <div class="user-nav-info">
                    <span>👤 Logged in as: <strong>{user['username']}</strong> ({user['email']})</span>
                    <span class="user-nav-badge">{role} · Authenticated</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_col2:
        if st.button("🚪 Log Out", key="logout_btn", use_container_width=True):
            logout_user()

    with st.expander("📱 Mobile Access QR Code (Scan with NeoReader or Phone Camera)", expanded=False):
        qr_c1, qr_c2 = st.columns([1, 3])
        with qr_c1:
            qr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode_network.png")
            if os.path.isfile(qr_path):
                st.image(qr_path, width=160, caption="Scan with NeoReader")
        with qr_c2:
            st.markdown(
                """
                **Open on your Smartphone:**
                - **Direct Wi-Fi URL:** `http://10.65.126.214:8501`
                - **NeoReader Scheme:** `neoreader://10.65.126.214:8501`

                *Ensure your smartphone is on the same local network / Wi-Fi.*
                """
            )


def require_authentication(background_image: str | None = None) -> bool:
    """
    Main authentication guard. Call at top of app.py.
    Returns True if authenticated, otherwise renders login and returns False.
    """
    if is_authenticated():
        return True
    render_auth_ui(background_image)
    return False
