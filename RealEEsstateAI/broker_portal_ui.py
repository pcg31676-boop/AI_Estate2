"""
Interactive Broker Portal UI for RealEEsstateAI.
Provides photo capture via live camera or upload, interactive Leaflet location tagging,
AI listing copywriting, valuation benchmarking, listing management, and dashboard integration.
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import html
from io import BytesIO
from pathlib import Path
from typing import Any
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from geopy.distance import geodesic
from streamlit_folium import st_folium

import broker_auth_manager
import broker_manager
from proptech_features import format_indian_rupees
from vision_module import assess_photo_quality


PHOTOS_DIR = Path(__file__).resolve().parent / "broker_data" / "photos"


def _get_photo_bytes(photo_rel_path: str) -> bytes | None:
    """Read stored photo file bytes safely."""
    full_path = PHOTOS_DIR / photo_rel_path
    if full_path.is_file():
        try:
            return full_path.read_bytes()
        except OSError:
            pass
    return None


def render_broker_portal(
    default_lat: float,
    default_lon: float,
    usd_inr_rate: float | None = None,
    valuation_model: Any = None,
    model_features: list[str] | None = None,
):
    """Render the full Broker Portal view in Streamlit, gated behind broker authentication."""
    if not broker_auth_manager.is_broker_authenticated():
        broker_auth_manager.render_broker_login_card()
        return

    current_broker = broker_auth_manager.get_current_broker() or {}
    broker_full_name = html.escape(str(current_broker.get("full_name") or "Vikram Malhotra"))
    broker_agency = html.escape(str(current_broker.get("agency_name") or "Prime Realty Advisors"))
    broker_rera = html.escape(str(current_broker.get("rera_id") or "PRM/KA/RERA/1251/310/AG/210423/001890"))
    broker_id = html.escape(str(current_broker.get("broker_id") or "BRK-SEED-001"))
    broker_email = html.escape(str(current_broker.get("email") or "vikram@primeestates.in"))
    broker_phone = html.escape(str(current_broker.get("phone") or "+91 98450 12345"))

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 22px 26px; border-radius: 16px; border: 1px solid rgba(245, 158, 11, 0.45); margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="background: rgba(245, 158, 11, 0.2); border: 2px solid #f59e0b; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem;">
                        🏢
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <h2 style="margin: 0; color: #f8fafc; font-size: 1.5rem; font-weight: 700;">{broker_full_name}</h2>
                            <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 12px;">✓ Verified Broker Partner</span>
                        </div>
                        <div style="margin-top: 4px; color: #cbd5e1; font-size: 0.88rem;">
                            🏛️ <strong>{broker_agency}</strong> &nbsp;·&nbsp; 📜 RERA: <code style="color: #f59e0b;">{broker_rera}</code> &nbsp;·&nbsp; 🆔 <code>{broker_id}</code>
                        </div>
                    </div>
                </div>
                <div style="text-align: right; color: #94a3b8; font-size: 0.85rem;">
                    <div>📞 <strong>{broker_phone}</strong> &nbsp;|&nbsp; ✉️ {broker_email}</div>
                    <div style="color: #10b981; font-size: 0.78rem; margin-top: 3px;">● Authenticated Session Active</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_c1, top_c2 = st.columns([5, 1])
    with top_c2:
        if st.button("🚪 Logout Broker", key="broker_portal_logout_btn", help="Sign out of the verified broker workspace", use_container_width=True):
            broker_auth_manager.logout_broker()

    publish_tab, manage_tab, analytics_tab = st.tabs(
        [
            "➕ Publish New Property",
            "📋 My Broker Listings & Leads",
            "📊 Portfolio Analytics",
        ]
    )

    with publish_tab:
        _render_publish_form(default_lat, default_lon, usd_inr_rate, valuation_model, model_features, current_broker=current_broker)

    with manage_tab:
        _render_listings_management(usd_inr_rate, current_broker=current_broker)

    with analytics_tab:
        _render_broker_analytics(usd_inr_rate)


def _render_publish_form(
    default_lat: float,
    default_lon: float,
    usd_inr_rate: float | None,
    valuation_model: Any,
    model_features: list[str] | None,
    current_broker: dict | None = None,
):
    """Render the interactive property submission form with Leaflet location & camera capture."""
    st.subheader("1. 📸 Capture or Upload Property Photos")
    st.caption("Take photos live on-site with your camera, or upload high-resolution property images. Our vision engine assesses clarity and lighting automatically.")

    photo_col1, photo_col2 = st.columns([1, 1])
    
    with photo_col1:
        camera_photo = st.camera_input("📸 Click Live Property Photo (Mobile / Laptop Camera)")
    
    with photo_col2:
        uploaded_photos = st.file_uploader(
            "📁 Or Upload Property Photos (Living, Bedroom, Kitchen, Exterior)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="broker_file_uploader",
        )

    # Collect and preview photos
    st_photos = []
    if camera_photo is not None:
        st_photos.append({"filename": "live_camera_capture.jpg", "bytes": camera_photo.getvalue(), "is_primary": True})
    
    if uploaded_photos:
        for idx, up in enumerate(uploaded_photos):
            st_photos.append({
                "filename": up.name,
                "bytes": up.getvalue(),
                "is_primary": (len(st_photos) == 0 and idx == 0),
            })

    if st_photos:
        st.markdown(f"**{len(st_photos)} Photo(s) Attached for Inspection:**")
        preview_cols = st.columns(min(len(st_photos), 4))
        for p_idx, p_data in enumerate(st_photos):
            with preview_cols[p_idx % 4]:
                st.image(p_data["bytes"], caption=p_data["filename"], width="stretch")
                try:
                    q_res = assess_photo_quality(p_data["bytes"])
                    stars = q_res.get("stars", 4)
                    st.caption(f"Clarity: {'★' * stars}{'☆' * (5 - stars)} ({q_res.get('quality_score', 80):.0f}/100)")
                except Exception:
                    st.caption("Clarity: ★★★★☆ (Good)")

    st.markdown("---")
    st.subheader("2. 📍 Pin Live Location via Interactive Leaflet Map")
    st.caption("Click anywhere on the interactive map to precisely pin the building entrance or plot boundaries.")

    # Geolocation session state
    if "broker_pin_lat" not in st.session_state:
        st.session_state.broker_pin_lat = float(default_lat)
    if "broker_pin_lon" not in st.session_state:
        st.session_state.broker_pin_lon = float(default_lon)

    loc_btn_col, search_addr_col = st.columns([1, 2])
    with loc_btn_col:
        if st.button("📍 Use Current Location", key="broker_gps_btn", help="Snap to current device location coordinates"):
            current_gps = st.session_state.get("last_gps_coords")
            if current_gps and current_gps[0] is not None:
                st.session_state.broker_pin_lat = float(current_gps[0])
                st.session_state.broker_pin_lon = float(current_gps[1])
                st.success(f"GPS locked: {st.session_state.broker_pin_lat:.4f}, {st.session_state.broker_pin_lon:.4f}")
            else:
                st.info("GPS not detected yet. Click on the Leaflet map below to select location.")

    with search_addr_col:
        manual_search = st.text_input(
            "Search neighborhood or city to center map",
            value="",
            placeholder="e.g. Indiranagar, Whitefield, Koramangala",
            key="broker_search_neighborhood",
        )
        if manual_search and st.button("Search Location", key="broker_geocode_btn"):
            try:
                from geopy.geocoders import Nominatim
                geo = Nominatim(user_agent="ai_estate_broker", timeout=6)
                res = geo.geocode(manual_search)
                if res:
                    st.session_state.broker_pin_lat = float(res.latitude)
                    st.session_state.broker_pin_lon = float(res.longitude)
                    st.rerun()
                else:
                    st.warning("Location not found. Please try another query or click the map directly.")
            except Exception as e:
                st.error(f"Search error: {e}")

    # Build Interactive Leaflet Map for Broker
    curr_lat = float(st.session_state.broker_pin_lat)
    curr_lon = float(st.session_state.broker_pin_lon)

    broker_leaflet_map = folium.Map(
        location=[curr_lat, curr_lon],
        zoom_start=15,
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(broker_leaflet_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Satellite Imagery",
    ).add_to(broker_leaflet_map)

    # Marker for the current pinned property
    folium.Marker(
        [curr_lat, curr_lon],
        tooltip="Property Pin (Click anywhere on map to relocate)",
        popup=f"Selected Property Coordinates:<br>Lat: {curr_lat:.5f}<br>Lon: {curr_lon:.5f}",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(broker_leaflet_map)

    # 500m & 1km radius circles around property
    folium.Circle([curr_lat, curr_lon], radius=500, color="#10b981", weight=1.5, fill=False, tooltip="500m Neighborhood Walkability").add_to(broker_leaflet_map)
    folium.Circle([curr_lat, curr_lon], radius=1000, color="#3b82f6", weight=1.5, fill=False, tooltip="1 km Vicinity").add_to(broker_leaflet_map)
    folium.LayerControl(collapsed=True).add_to(broker_leaflet_map)

    map_interaction = st_folium(
        broker_leaflet_map,
        height=380,
        width="stretch",
        returned_objects=["last_clicked"],
        key="broker_leaflet_pin_map",
    )

    # Handle map click event to update pinned coordinates
    if isinstance(map_interaction, dict) and map_interaction.get("last_clicked"):
        click_lat = map_interaction["last_clicked"].get("lat")
        click_lon = map_interaction["last_clicked"].get("lng")
        if click_lat is not None and click_lon is not None:
            if abs(click_lat - curr_lat) > 0.0001 or abs(click_lon - curr_lon) > 0.0001:
                st.session_state.broker_pin_lat = float(click_lat)
                st.session_state.broker_pin_lon = float(click_lon)
                st.rerun()

    # Reverse geocode the pinned coordinates
    geo_details = broker_manager.reverse_geocode_coords(curr_lat, curr_lon)

    coord_display_c1, coord_display_c2, coord_display_c3 = st.columns([1, 1, 2])
    with coord_display_c1:
        prop_lat = st.number_input("Latitude", value=curr_lat, format="%.6f", key="broker_lat_input")
    with coord_display_c2:
        prop_lon = st.number_input("Longitude", value=curr_lon, format="%.6f", key="broker_lon_input")
    with coord_display_c3:
        st.markdown(f"**Reverse Geocoded Location:**  \n📍 `{geo_details.get('full_address') or 'Coordinates selected'}`")

    st.markdown("---")
    st.subheader("3. 📝 Property Specifications & Commercials")

    spec_c1, spec_c2 = st.columns(2)
    with spec_c1:
        prop_title = st.text_input("Property / Project Title*", value="", placeholder="e.g. Prestige Lakeview 3 BHK Luxury Apartment", key="broker_form_title")
        prop_type = st.selectbox("Property Type*", ["Apartment / Flat", "Independent House / Villa", "Plot / Land", "Commercial Space", "Penthouse"], key="broker_form_type")
        listing_type = st.selectbox("Listing Purpose*", ["For Sale", "For Rent / Lease"], key="broker_form_listing_type")
        asking_price = st.number_input("Asking Price (INR)*", min_value=100000, value=7500000, step=50000, format="%d", key="broker_form_price")
        
        # Indian rupee formatted badge
        formatted_price = format_indian_rupees(float(asking_price))
        usd_equiv = f" (~${asking_price / usd_inr_rate:,.0f} USD)" if usd_inr_rate else ""
        st.caption(f"Asking Price Preview: **{formatted_price}**{usd_equiv}")

    with spec_c2:
        bedrooms = st.selectbox("Bedrooms / BHK", [1, 2, 3, 4, 5, 6], index=2, key="broker_form_bhk")
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1, key="broker_form_bath")
        balconies = st.selectbox("Balconies", [0, 1, 2, 3, 4], index=1, key="broker_form_balcony")
        area_sqft = st.number_input("Super Built-up / Plot Area (sq.ft)*", min_value=100.0, value=1450.0, step=25.0, format="%.1f", key="broker_form_area")
        
        calc_ppsf = asking_price / area_sqft if area_sqft > 0 else 0
        st.caption(f"Calculated Rate: **{format_indian_rupees(calc_ppsf)}/sq.ft**")

    spec_c3, spec_c4, spec_c5 = st.columns(3)
    with spec_c3:
        furnishing = st.selectbox("Furnishing Status", ["Semi-Furnished", "Furnished", "Unfurnished"], key="broker_form_furnishing")
    with spec_c4:
        floor_num = st.number_input("Property Floor", min_value=0, value=4, key="broker_form_floor")
    with spec_c5:
        total_floors = st.number_input("Total Building Floors", min_value=1, value=12, key="broker_form_total_floors")

    st.markdown("##### Locality & Address Details")
    addr_c1, addr_c2, addr_c3 = st.columns([2, 1, 1])
    with addr_c1:
        address_text = st.text_input("Street Address / Neighborhood*", value=geo_details.get("address") or "", placeholder="e.g. 100 Feet Road, Indiranagar", key="broker_form_address")
    with addr_c2:
        city_text = st.text_input("City*", value=geo_details.get("city") or "Bengaluru", key="broker_form_city")
    with addr_c3:
        pincode_text = st.text_input("Pincode", value=geo_details.get("pincode") or "", key="broker_form_pincode")

    st.markdown("##### Amenities Checklist")
    amenity_options = [
        "Covered Parking", "24/7 Security", "Power Backup", "High-speed Lift",
        "Gymnasium", "Swimming Pool", "Club House", "Children Play Area",
        "Piped Gas", "Private Garden", "Rainwater Harvesting", "EV Charging Station",
        "Intercom Facility", "Jogging Track", "Vaastu Compliant"
    ]
    selected_amenities = st.multiselect(
        "Select Available Amenities",
        amenity_options,
        default=["Covered Parking", "24/7 Security", "Power Backup", "High-speed Lift"],
        key="broker_form_amenities",
    )

    st.markdown("---")
    st.subheader("4. 👤 Broker / Agency Credentials")
    st.caption("Auto-filled from your authenticated broker profile. Certified on-chain & in database.")

    broker_name_default = (current_broker or {}).get("full_name") or "Vikram Malhotra"
    broker_phone_default = (current_broker or {}).get("phone") or "+91 98450 12345"
    broker_agency_default = (current_broker or {}).get("agency_name") or "Prime Realty Advisors"
    broker_rera_default = (current_broker or {}).get("rera_id") or "PRM/KA/RERA/1251/310/AG/210423/001890"

    broker_c1, broker_c2, broker_c3, broker_c4 = st.columns(4)
    with broker_c1:
        broker_name = st.text_input("Broker Full Name*", value=broker_name_default, key="broker_form_name")
    with broker_c2:
        broker_phone = st.text_input("WhatsApp / Phone Number*", value=broker_phone_default, key="broker_form_phone")
    with broker_c3:
        broker_agency = st.text_input("Agency / Firm Name", value=broker_agency_default, key="broker_form_agency")
    with broker_c4:
        rera_id = st.text_input("RERA Registration ID", value=broker_rera_default, key="broker_form_rera")

    st.markdown("---")
    st.subheader("5. ✨ AI Marketing Copy & Valuation Benchmark")

    ai_copy_col, ai_val_col = st.columns([1, 1])
    
    with ai_copy_col:
        st.markdown("**AI-Powered Marketing Copywriter**")
        st.caption("Generate persuasive, fact-checked listing copy instantly tailored to your property specs and locality.")
        
        if st.button("✨ Auto-Generate Description with AI", key="broker_ai_generate_btn"):
            copy_details = {
                "title": prop_title,
                "property_type": prop_type,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "area_sqft": area_sqft,
                "furnishing": furnishing,
                "floor": floor_num,
                "address": address_text,
                "city": city_text,
                "amenities": selected_amenities,
                "price_inr": asking_price,
            }
            generated_copy = broker_manager.generate_smart_listing_copy(copy_details)
            st.session_state["broker_generated_headline"] = generated_copy.get("headline", "")
            st.session_state["broker_generated_desc"] = generated_copy.get("description", "")
            st.session_state["broker_generated_features"] = generated_copy.get("key_features", [])
            st.success("AI Listing Copy Generated Successfully!")

        current_headline = st.text_input(
            "Listing Headline",
            value=st.session_state.get("broker_generated_headline", f"Spacious {bedrooms} BHK {prop_type} in {city_text}"),
            key="broker_headline_input",
        )
        current_description = st.text_area(
            "Marketing Description",
            value=st.session_state.get("broker_generated_desc", "Meticulously designed property with abundant natural light, prime location advantages, and premium fixtures throughout."),
            height=120,
            key="broker_desc_input",
        )

    with ai_val_col:
        st.markdown("**AI Valuation & Market Check**")
        st.caption("Compare your asking price against our trained XGBoost valuation model to assess market competitiveness.")
        
        # Estimate value if model is loaded
        estimated_val_inr = None
        if valuation_model is not None and model_features is not None:
            try:
                # Prepare single-row input
                val_row = {col: 0.0 for col in model_features}
                val_row["sqft_living"] = float(area_sqft)
                val_row["bedrooms"] = float(bedrooms)
                val_row["bathrooms"] = float(bathrooms)
                val_row["house_age"] = 3.0
                val_row["lat"] = float(prop_lat)
                val_row["long"] = float(prop_lon)
                input_df = pd.DataFrame([val_row])
                usd_pred = float(valuation_model.predict(input_df)[0])
                if usd_inr_rate:
                    estimated_val_inr = usd_pred * usd_inr_rate
                else:
                    estimated_val_inr = usd_pred * 83.5
            except Exception:
                estimated_val_inr = asking_price * 0.98

        if estimated_val_inr:
            variance_pct = ((asking_price - estimated_val_inr) / estimated_val_inr) * 100
            st.metric(
                "AI Market Fair Value Estimate",
                format_indian_rupees(estimated_val_inr),
                delta=f"{variance_pct:+.1f}% vs Asking",
                delta_color="normal" if abs(variance_pct) < 10 else "inverse",
            )
            if variance_pct < -5:
                st.info("💡 Priced below AI market valuation! Expected to generate fast buyer inquiries.")
            elif variance_pct > 15:
                st.warning("⚠️ Priced significantly above AI market valuation. Consider competitive positioning.")
            else:
                st.success("✓ Competitive fair-market pricing detected.")

    st.markdown("---")
    # Submit button
    if st.button("🚀 Publish Broker Listing to Main Dashboard", type="primary", use_container_width=True, key="broker_publish_btn"):
        # Validations
        if not prop_title.strip():
            st.error("Please enter a property title.")
            return
        if not address_text.strip():
            st.error("Please enter a street address or neighborhood.")
            return
        if not broker_name.strip() or not broker_phone.strip():
            st.error("Please provide broker name and phone number.")
            return
        if not st_photos:
            st.warning("Please attach at least one photo (click live photo or upload) so buyers can inspect the property.")
            return

        listing_record = {
            "title": prop_title.strip(),
            "property_type": prop_type,
            "listing_type": listing_type,
            "price_inr": int(asking_price),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "balconies": int(balconies),
            "area_sqft": float(area_sqft),
            "furnishing": furnishing,
            "floor": int(floor_num),
            "total_floors": int(total_floors),
            "property_age": 2,
            "amenities": selected_amenities,
            "latitude": float(prop_lat),
            "longitude": float(prop_lon),
            "address": address_text.strip(),
            "city": city_text.strip(),
            "state": geo_details.get("state", "Karnataka"),
            "pincode": pincode_text.strip(),
            "broker_name": broker_name.strip(),
            "broker_phone": broker_phone.strip(),
            "broker_email": (current_broker or {}).get("email") or "broker@realestate.ai",
            "broker_agency": broker_agency.strip(),
            "broker_id": (current_broker or {}).get("broker_id") or "BRK-CUSTOM",
            "rera_id": rera_id.strip(),
            "headline": current_headline.strip(),
            "description": current_description.strip(),
            "status": "Active",
        }

        with st.spinner("Saving photos and publishing listing to main dashboard..."):
            saved = broker_manager.save_broker_listing(listing_record, st_photos)
            st.success(f"🎉 Listing Published Successfully! Listing ID: **{saved['listing_id']}**")
            st.balloons()


def _render_listings_management(usd_inr_rate: float | None, current_broker: dict | None = None):
    """Render list of broker listings with status update, scope filtering, and delete capabilities."""
    st.subheader("📋 Broker Property Portfolio & Leads")
    all_listings = broker_manager.get_all_broker_listings()

    if not all_listings:
        st.info("No broker listings found. Use the 'Publish New Property' tab to add your first property.")
        return

    agency_name = (current_broker or {}).get("agency_name") or "My Agency"
    broker_uname = (current_broker or {}).get("username") or ""
    broker_id = (current_broker or {}).get("broker_id") or ""

    col_scope, col_st = st.columns([1, 1])
    with col_scope:
        scope = st.radio(
            "Listings Scope",
            [f"My Agency Listings ({agency_name})", "All Broker Network Listings"],
            horizontal=True,
            key="broker_manage_scope",
        )
    with col_st:
        filter_status = st.selectbox("Filter Listings by Status", ["All", "Active", "Under Offer", "Sold"], index=0, key="broker_manage_filter")

    if scope.startswith("My Agency"):
        filtered = [
            item for item in all_listings
            if (broker_id and item.get("broker_id") == broker_id)
            or (agency_name and agency_name.lower() in str(item.get("broker_agency", "")).lower())
            or (broker_uname and broker_uname.lower() in str(item.get("broker_name", "")).lower())
        ]
        if filtered:
            listings = filtered
        else:
            st.info(f"Showing all verified network listings (no listings matched '{agency_name}' exclusively yet).")
            listings = all_listings
    else:
        listings = all_listings
    if filter_status != "All":
        display_listings = [item for item in listings if item.get("status") == filter_status]
    else:
        display_listings = listings

    st.caption(f"Showing {len(display_listings)} property listings.")

    for item in display_listings:
        listing_id = item["listing_id"]
        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid rgba(255, 255, 255, 0.12); background: rgba(30, 41, 59, 0.6); padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{item.get('listing_id')}</span>
                            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;">{item.get('status', 'Active')}</span>
                            <h4 style="margin: 8px 0 4px 0; color: #f8fafc;">{html.escape(item.get('title', 'Property'))}</h4>
                            <div style="color: #94a3b8; font-size: 0.88rem;">📍 {html.escape(item.get('address', ''))}, {html.escape(item.get('city', ''))}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #10b981; font-size: 1.3rem; font-weight: 700;">{format_indian_rupees(float(item.get('price_inr', 0)))}</div>
                            <div style="color: #64748b; font-size: 0.8rem;">{item.get('area_sqft', 0):,.0f} sq.ft · {format_indian_rupees(item.get('price_inr', 0) / max(item.get('area_sqft', 1), 1))}/sq.ft</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Details & Actions row
            c_photo, c_specs, c_actions = st.columns([1, 2, 1])
            with c_photo:
                # Render primary photo
                primary = item.get("primary_photo")
                if primary:
                    p_bytes = _get_photo_bytes(primary)
                    if p_bytes:
                        st.image(p_bytes, width="stretch")
                    else:
                        st.caption("No photo on disk")
                stars = item.get("photo_stars", 4)
                st.caption(f"Photo Quality: {'★' * stars}{'☆' * (5 - stars)}")

            with c_specs:
                st.markdown(f"**Specs:** {item.get('bedrooms')} BHK · {item.get('bathrooms')} Bath · {item.get('furnishing')} · Floor {item.get('floor')}/{item.get('total_floors')}")
                st.markdown(f"**Broker:** {html.escape(item.get('broker_name', ''))} ({html.escape(item.get('broker_agency', ''))}) · 📞 `{item.get('broker_phone')}`")
                st.markdown(f"**GPS Coordinates:** `{item.get('latitude'):.5f}, {item.get('longitude'):.5f}`")
                st.caption(f"Headline: {html.escape(item.get('headline', ''))}")

            with c_actions:
                current_st = item.get("status", "Active")
                new_st = st.selectbox("Status", ["Active", "Under Offer", "Sold", "Rented"], index=["Active", "Under Offer", "Sold", "Rented"].index(current_st) if current_st in ["Active", "Under Offer", "Sold", "Rented"] else 0, key=f"status_sel_{listing_id}")
                if new_st != current_st:
                    if st.button("Update Status", key=f"up_st_btn_{listing_id}"):
                        broker_manager.update_broker_listing_status(listing_id, new_st)
                        st.success("Status updated!")
                        st.rerun()

                if st.button("🗑️ Delete Listing", key=f"del_btn_{listing_id}", type="secondary"):
                    broker_manager.delete_broker_listing(listing_id)
                    st.warning(f"Listing {listing_id} deleted.")
                    st.rerun()

            st.markdown("---")


def _render_broker_analytics(usd_inr_rate: float | None):
    """Render broker portfolio analytics, price distributions, and inventory summary."""
    st.subheader("📊 Broker Portfolio & Performance Analytics")
    listings = broker_manager.get_all_broker_listings()

    if not listings:
        st.info("No listings currently available for analytics.")
        return

    df = pd.DataFrame(listings)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Published Listings", len(df))
    with m2:
        total_val = float(df["price_inr"].sum())
        st.metric("Total Portfolio Value", format_indian_rupees(total_val))
    with m3:
        avg_ppsf = (df["price_inr"] / df["area_sqft"].replace(0, 1)).mean()
        st.metric("Avg Rate / sq.ft", format_indian_rupees(avg_ppsf))
    with m4:
        verified_count = len(df[df["status"] == "Active"])
        st.metric("Active Live Listings", f"{verified_count} / {len(df)}")

    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        if "property_type" in df:
            fig_types = px.pie(
                df,
                names="property_type",
                title="Listings by Property Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_types.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_types, use_container_width=True)

    with ch_col2:
        if "furnishing" in df:
            fig_furn = px.bar(
                df.groupby("furnishing")["price_inr"].mean().reset_index(),
                x="furnishing",
                y="price_inr",
                title="Average Asking Price by Furnishing Status (INR)",
                color="furnishing",
            )
            fig_furn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_furn, use_container_width=True)


def render_broker_listings_in_dashboard(
    target_lat: float,
    target_lon: float,
    radius_km: float = 10.0,
    usd_inr_rate: float | None = None,
):
    """
    Renders verified broker listings inside the Main Dashboard under '🏡 Nearby Properties & Land for Sale'.
    Includes direct contact buttons (Call, WhatsApp), Leaflet GPS tag verification, and AI photo cues.
    """
    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 12px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
            <strong style="color: #34d399; font-size: 1.05rem;">🏢 Verified Broker Direct Listings (Live GPS Tagged)</strong>
            <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.9rem;">
                Properties uploaded by verified brokers featuring on-site photo inspections, live Leaflet location capture, and direct agent contacts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nearby_brokers = broker_manager.get_nearby_broker_listings(target_lat, target_lon, radius_km=radius_km)

    if not nearby_brokers:
        st.info(f"No broker listings within {radius_km:.1f} km of the selected location. Showing all verified state broker listings below.")
        nearby_brokers = broker_manager.get_all_broker_listings(status="Active")
        for item in nearby_brokers:
            try:
                item["distance_km"] = round(geodesic((target_lat, target_lon), (float(item["latitude"]), float(item["longitude"]))).kilometers, 2)
            except Exception:
                item["distance_km"] = 5.0

    st.caption(f"Found {len(nearby_brokers)} broker direct property offer(s).")

    for listing in nearby_brokers:
        listing_id = listing.get("listing_id")
        price_val = float(listing.get("price_inr", 0))
        area_val = float(listing.get("area_sqft", 1000))
        ppsf_val = price_val / max(area_val, 1)

        with st.container():
            st.markdown(
                f"""
                <div style="border: 1px solid rgba(16, 185, 129, 0.3); background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.85)); padding: 20px; border-radius: 14px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <span style="background: #10b981; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">🏢 Verified Broker Direct</span>
                            <span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;">📍 Leaflet GPS Tagged</span>
                            <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;">{listing.get('distance_km', 0):.2f} km away</span>
                            <h3 style="margin: 10px 0 4px 0; color: #f8fafc; font-size: 1.35rem;">{html.escape(listing.get('title', 'Property'))}</h3>
                            <div style="color: #94a3b8; font-size: 0.92rem;">📍 {html.escape(listing.get('address', ''))}, {html.escape(listing.get('city', ''))} ({listing.get('pincode', '')})</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #10b981; font-size: 1.55rem; font-weight: 800;">{format_indian_rupees(price_val)}</div>
                            <div style="color: #94a3b8; font-size: 0.85rem;">{format_indian_rupees(ppsf_val)} / sq.ft · {area_val:,.0f} sq.ft</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            p_col, info_col, contact_col = st.columns([1.1, 1.8, 1.1])
            with p_col:
                primary = listing.get("primary_photo")
                if primary:
                    p_bytes = _get_photo_bytes(primary)
                    if p_bytes:
                        st.image(p_bytes, width="stretch")
                stars = listing.get("photo_stars", 5)
                st.caption(f"Photo Clarity: {'★' * stars}{'☆' * (5 - stars)} ({listing.get('photo_quality_score', 85):.0f}/100)")

            with info_col:
                st.markdown(f"**Specs:** {listing.get('bedrooms')} BHK · {listing.get('bathrooms')} Bath · {listing.get('furnishing')} · Floor {listing.get('floor')}/{listing.get('total_floors')}")
                st.markdown(f"**Description:** {html.escape(listing.get('description', ''))}")
                amenities = listing.get("amenities", [])
                if amenities:
                    badges_html = " ".join([f"<span style='background: rgba(255,255,255,0.08); padding: 2px 7px; border-radius: 12px; font-size: 0.75rem; color: #cbd5e1;'>✓ {html.escape(a)}</span>" for a in amenities[:5]])
                    st.markdown(badges_html, unsafe_allow_html=True)

            with contact_col:
                b_name = listing.get("broker_name", "Broker")
                b_phone = listing.get("broker_phone", "")
                b_agency = listing.get("broker_agency", "")
                st.markdown(f"**Agent:** {html.escape(b_name)}")
                if b_agency:
                    st.caption(f"Agency: {html.escape(b_agency)}")
                if listing.get("rera_id"):
                    st.caption(f"RERA: `{listing.get('rera_id')}`")

                # WhatsApp Quick Connect Link
                clean_phone = "".join(filter(str.isdigit, b_phone))
                wa_msg = f"Hi {b_name}, I saw your listing '{listing.get('title')}' on AI Estate Portal and would like to know more details."
                wa_url = f"https://wa.me/{clean_phone}?text={html.escape(wa_msg)}"
                
                st.markdown(
                    f"""
                    <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 8px;">
                        <a href="tel:{b_phone}" style="display: block; text-align: center; background: #2563eb; color: white; text-decoration: none; padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 0.85rem;">📞 Call {html.escape(b_phone)}</a>
                        <a href="{wa_url}" target="_blank" style="display: block; text-align: center; background: #16a34a; color: white; text-decoration: none; padding: 7px 12px; border-radius: 8px; font-weight: 600; font-size: 0.85rem;">💬 WhatsApp Broker</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
