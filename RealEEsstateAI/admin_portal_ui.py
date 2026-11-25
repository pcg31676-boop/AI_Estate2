"""Administrator dashboard for account oversight and broker listing moderation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import auth_manager
import broker_auth_manager
import broker_manager


def render_admin_portal() -> None:
    """Render the authenticated admin interface."""
    admin = auth_manager.get_current_user() or {}
    if admin.get("role") != "admin":
        st.error("Administrator access is required to view this page.")
        return

    st.title("🛡️ AI Estate Admin")
    st.caption(f"Signed in as {admin.get('username')} · Platform administration")

    users = auth_manager.list_registered_users()
    brokers = broker_auth_manager.list_registered_brokers()
    listings = broker_manager.get_all_broker_listings()

    metric_user, metric_broker, metric_listing = st.columns(3)
    metric_user.metric("Registered accounts", len(users))
    metric_broker.metric("Broker partners", len(brokers))
    metric_listing.metric("Property listings", len(listings))

    users_tab, brokers_tab, listings_tab = st.tabs(
        ["👤 User Accounts", "🏢 Broker Accounts", "🏘️ Property Listings"]
    )

    with users_tab:
        st.subheader("Registered user and admin accounts")
        if users:
            st.dataframe(
                pd.DataFrame(users),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No user accounts have been registered yet.")

    with brokers_tab:
        st.subheader("Broker partner directory")
        if brokers:
            st.dataframe(
                pd.DataFrame(brokers),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No broker partner accounts are registered yet.")

    with listings_tab:
        st.subheader("Review and moderate broker listings")
        if not listings:
            st.info("There are no broker property listings to review.")
        else:
            listing_options = {
                f"{item.get('title', 'Untitled')} · {item.get('listing_id')}": item
                for item in listings
            }
            selected_label = st.selectbox(
                "Select a property listing",
                list(listing_options),
                key="admin_listing_selection",
            )
            selected_listing = listing_options[selected_label]
            current_status = selected_listing.get("status", "Active")
            status_options = ["Active", "Under Offer", "Sold", "Rented", "Rejected"]
            if current_status not in status_options:
                status_options.append(current_status)
            next_status = st.selectbox(
                "Listing status",
                status_options,
                index=status_options.index(current_status),
                key=f"admin_listing_status_{selected_listing.get('listing_id')}",
            )
            st.write(
                f"**{selected_listing.get('title', 'Untitled')}** · "
                f"{selected_listing.get('city', 'Location unavailable')} · "
                f"{selected_listing.get('broker_name', 'Broker unavailable')}"
            )
            if st.button(
                "Save listing status",
                type="primary",
                use_container_width=True,
                key="admin_listing_status_save",
            ):
                if broker_manager.update_broker_listing_status(
                    str(selected_listing["listing_id"]), next_status
                ):
                    st.success("Listing status updated.")
                    st.rerun()
                else:
                    st.error("The listing could not be updated. Refresh and try again.")

    st.divider()
    if st.button("🚪 Sign out", key="admin_logout", use_container_width=True):
        auth_manager.logout_user()
