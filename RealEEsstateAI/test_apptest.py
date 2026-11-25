"""
End-to-End Streamlit App Simulation using Streamlit AppTest.
Verifies authentication guard, dashboard unlock, and photo analysis tab rendering.
"""

from streamlit.testing.v1 import AppTest

print("--- [1] Testing Unauthenticated Session Gate ---")
at = AppTest.from_file("app.py", default_timeout=30)
at.run()
assert not at.exception, f"Unauthenticated session threw exception: {at.exception}"
# In unauthenticated state, login card should be rendered
login_inputs = [inp for inp in at.text_input if inp.key in {"login_identifier", "reg_username"}]
assert len(login_inputs) >= 1, "Login/Registration inputs must be rendered for unauthenticated session"
login_tab_labels = [tab.label for tab in at.tabs]
assert "👤 User Sign In" in login_tab_labels, "User login interface must be available"
assert "🛡️ Admin Sign In" in login_tab_labels, "Admin login interface must be available"
print(f"✔ Login gate active: {len(at.text_input)} text inputs rendered, unauthenticated session safely stopped at gate.")

print("\n--- [2] Testing Authenticated Session & Dashboard Unlock ---")
at_auth = AppTest.from_file("app.py", default_timeout=45)
at_auth.session_state["authenticated"] = True
at_auth.session_state["auth_user"] = {
    "id": 1,
    "username": "tester",
    "email": "tester@example.com",
    "last_login": "2026-11-16T22:20:00"
}
at_auth.run()
assert not at_auth.exception, f"Authenticated session threw exception: {at_auth.exception}"

# Verify user navbar rendered
logout_buttons = [btn for btn in at_auth.button if btn.key == "logout_btn"]
assert len(logout_buttons) == 1, "Logout button must be rendered on user navbar"
print("✔ User navbar active with Logout button")

# Verify user dashboard tabs. Broker publishing is available on its dedicated
# Streamlit page rather than embedded in the user dashboard.
tabs = at_auth.tabs
tab_labels = [t.label for t in tabs]
print(f"✔ Dashboard tabs rendered ({len(tabs)} tabs):")
for label in tab_labels:
    print(f"   - {label}")

assert "📸 AI Photo Analysis" in tab_labels, "📸 AI Photo Analysis tab must be present"
assert "Valuation" in tab_labels, "Valuation tab must be present"
assert "🏡 Nearby Properties & Land for Sale" in tab_labels, "Nearby Properties tab must be present"
assert "AI Studio" in tab_labels, "AI Studio tab must be present"
assert not any("Broker Portal" in label for label in tab_labels), "Broker Portal must remain a separate page"

print("\n--- [3] Testing Admin Role Dashboard ---")
at_admin = AppTest.from_file("app.py", default_timeout=45)
at_admin.session_state["authenticated"] = True
at_admin.session_state["auth_user"] = {
    "id": 2,
    "username": "admin_test",
    "email": "admin_test@example.com",
    "role": "admin",
}
at_admin.run()
assert not at_admin.exception, f"Admin session threw exception: {at_admin.exception}"
assert any("AI Estate Admin" in item.value for item in at_admin.title), "Admin dashboard must render"
admin_tab_labels = [tab.label for tab in at_admin.tabs]
assert "👤 User Accounts" in admin_tab_labels, "Admin account overview must render"
assert "Valuation" not in admin_tab_labels, "Admin users must not be sent into the user dashboard"

print("\n--- [4] Testing Dedicated Broker Portal Page ---")
at_broker_page = AppTest.from_file("pages/1_Broker_Portal.py", default_timeout=45)
at_broker_page.run()
assert not at_broker_page.exception, f"Broker page threw exception: {at_broker_page.exception}"
assert any(
    inp.key == "broker_auth_identifier" for inp in at_broker_page.text_input
), "Dedicated broker page must render broker sign-in"

print("\n🎉 ALL STREAMLIT APP TESTS PASSED WITH 100% SUCCESS!")
