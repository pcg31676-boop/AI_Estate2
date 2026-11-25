"""Investment, locality, and PDF-report helpers for AI Estate."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from html import escape as html_escape
import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import simpleSplit
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as pdf_canvas
from xgboost import DMatrix


AMENITY_CATEGORIES = {
    "Schools": '[amenity~"^(school|college|university)$"]',
    "Hospitals": '[amenity~"^(hospital|clinic)$"]',
    "Public transit": (
        '[railway~"^(station|tram_stop)$"]'
        '[highway=bus_stop]'
        '[public_transport~"^(station|stop_position)$"]'
    ),
    "Commercial centers": (
        '[shop~"^(mall|supermarket)$"]'
        '[amenity=marketplace]'
    ),
    "Dining & nightlife": (
        '[amenity~"^(restaurant|cafe|bar|pub|fast_food|food_court)$"]'
    ),
    "Parks & green space": (
        '[leisure~"^(park|garden|nature_reserve)$"]'
        '[landuse=recreation_ground]'
    ),
    "Safety / walkability": (
        '[amenity~"^(police|fire_station)$"]'
        '[highway~"^(footway|pedestrian|path)$"]'
    ),
}


def fetch_usd_inr_rate() -> tuple[float, str]:
    """Fetch the latest published USD-to-INR rate and its UTC update time."""
    request = Request(
        "https://open.er-api.com/v6/latest/USD",
        headers={"User-Agent": "AI-Estate-Valuation/1.0"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not retrieve the USD-to-INR rate: {error}") from error

    if not isinstance(payload, dict) or payload.get("result") != "success":
        raise RuntimeError("The exchange-rate service returned an unsuccessful response.")
    try:
        rate = float(payload["rates"]["INR"])
        updated_at = datetime.fromtimestamp(
            int(payload["time_last_update_unix"]),
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M UTC")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("The exchange-rate service returned an invalid USD-to-INR rate.") from error
    if not math.isfinite(rate) or rate <= 0:
        raise RuntimeError("The exchange-rate service returned a non-positive USD-to-INR rate.")
    return rate, updated_at


def format_indian_rupees(amount: float) -> str:
    """Format an amount using Indian digit grouping and the rupee symbol."""
    if not math.isfinite(amount):
        raise ValueError("Currency amount must be finite.")
    rounded_amount = int(
        Decimal(str(abs(amount))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    digits = str(rounded_amount)
    if len(digits) > 3:
        last_three = digits[-3:]
        leading = digits[:-3]
        groups = []
        while leading:
            groups.append(leading[-2:])
            leading = leading[:-2]
        digits = ",".join(reversed(groups)) + "," + last_three
    sign = "-" if amount < 0 else ""
    return f"{sign}₹{digits}"


def calculate_investment_metrics(
    property_value: float,
    annual_rent_yield_pct: float,
    down_payment_pct: float,
    annual_interest_pct: float,
    loan_term_years: int,
    monthly_maintenance: float,
) -> dict[str, float]:
    """Estimate rent, cap rate, mortgage payment, and monthly cash flow in USD."""
    values = (
        property_value,
        annual_rent_yield_pct,
        down_payment_pct,
        annual_interest_pct,
        monthly_maintenance,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Investment inputs must be finite numbers.")
    if property_value <= 0 or loan_term_years <= 0 or monthly_maintenance < 0:
        raise ValueError("Property value, loan term, and maintenance must be valid.")
    if not 0 <= annual_rent_yield_pct <= 30:
        raise ValueError("Annual rent yield must be between 0% and 30%.")
    if not 0 <= down_payment_pct <= 100:
        raise ValueError("Down payment must be between 0% and 100%.")
    if not 0 <= annual_interest_pct <= 50:
        raise ValueError("Annual interest rate must be between 0% and 50%.")

    annual_rent = property_value * annual_rent_yield_pct / 100
    monthly_rent = annual_rent / 12
    annual_maintenance = monthly_maintenance * 12
    net_operating_income = annual_rent - annual_maintenance
    down_payment = property_value * down_payment_pct / 100
    loan_principal = property_value - down_payment
    monthly_rate = annual_interest_pct / 1200
    payments = loan_term_years * 12
    monthly_mortgage = (
        loan_principal / payments
        if monthly_rate == 0
        else loan_principal
        * monthly_rate
        / (1 - (1 + monthly_rate) ** -payments)
    )
    return {
        "monthly_rent": monthly_rent,
        "annual_gross_rent": annual_rent,
        "annual_maintenance": annual_maintenance,
        "net_operating_income": net_operating_income,
        "cap_rate_pct": net_operating_income / property_value * 100,
        "down_payment": down_payment,
        "loan_principal": loan_principal,
        "monthly_mortgage": monthly_mortgage,
        "monthly_cash_flow": monthly_rent - monthly_maintenance - monthly_mortgage,
    }


def calculate_micro_location_score(
    amenities: dict[str, dict[str, float | int | None]],
) -> float:
    """Convert nearby OSM amenity counts into a bounded heuristic score from 1 to 10."""
    category_weights = {
        "Schools": 1.0,
        "Public transit": 1.25,
        "Hospitals": 1.5,
        "Commercial centers": 1.0,
    }
    weighted_count = 0.0
    for category, weight in category_weights.items():
        category_details = amenities.get(category)
        if category_details is None:
            raise ValueError(f"Nearby amenity results are missing {category!r}.")
        try:
            count = int(category_details["count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Nearby amenity results contain an invalid count for {category!r}."
            ) from error
        if count < 0:
            raise ValueError("Nearby amenity counts cannot be negative.")
        weighted_count += count * weight
    return round(1 + 9 * (1 - math.exp(-weighted_count / 12)), 1)


def project_investment_cash_flows(
    *,
    property_value: float,
    annual_rent_yield_pct: float,
    annual_interest_pct: float,
    down_payment_pct: float,
    loan_term_years: int,
    annual_appreciation_pct: float,
    monthly_maintenance: float,
    projection_years: int = 10,
) -> list[dict[str, float | int | None]]:
    """Project annual rent, financing, equity, cash flow, and ROI in USD."""
    if projection_years <= 0:
        raise ValueError("Projection period must be positive.")
    if not 0 <= annual_appreciation_pct <= 30:
        raise ValueError("Annual appreciation must be between 0% and 30%.")
    metrics = calculate_investment_metrics(
        property_value=property_value,
        annual_rent_yield_pct=annual_rent_yield_pct,
        down_payment_pct=down_payment_pct,
        annual_interest_pct=annual_interest_pct,
        loan_term_years=loan_term_years,
        monthly_maintenance=monthly_maintenance,
    )
    monthly_rate = annual_interest_pct / 1200
    monthly_payment = metrics["monthly_mortgage"]
    balance = metrics["loan_principal"]
    initial_investment = metrics["down_payment"]
    cumulative_cash_flow = 0.0
    projection = []

    for year in range(1, projection_years + 1):
        property_value_at_year = property_value * (
            1 + annual_appreciation_pct / 100
        ) ** year
        gross_rent = (
            property_value_at_year * annual_rent_yield_pct / 100
        )
        maintenance = monthly_maintenance * 12
        annual_debt_service = 0.0
        for _ in range(12):
            if balance <= 0:
                break
            interest_payment = balance * monthly_rate
            principal_payment = min(balance, monthly_payment - interest_payment)
            balance -= principal_payment
            annual_debt_service += interest_payment + principal_payment

        net_cash_flow = gross_rent - maintenance - annual_debt_service
        cumulative_cash_flow += net_cash_flow
        equity = property_value_at_year - balance
        roi_pct = (
            (equity + cumulative_cash_flow - initial_investment)
            / initial_investment
            * 100
            if initial_investment > 0
            else None
        )
        projection.append(
            {
                "year": year,
                "gross_rent": gross_rent,
                "maintenance": maintenance,
                "debt_service": annual_debt_service,
                "net_cash_flow": net_cash_flow,
                "cumulative_cash_flow": cumulative_cash_flow,
                "property_value": property_value_at_year,
                "remaining_loan": balance,
                "equity": equity,
                "roi_pct": roi_pct,
            }
        )
    return projection


def fetch_nearby_amenities(
    latitude: float,
    longitude: float,
    radius_m: int = 1500,
) -> dict[str, dict[str, float | int | None]]:
    """Query Overpass and score nearby amenities by count and nearest distance."""
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude or longitude is outside its valid range.")
    if not 100 <= radius_m <= 5000:
        raise ValueError("Amenity search radius must be between 100 and 5,000 metres.")

    selectors = []
    for clauses in AMENITY_CATEGORIES.values():
        selectors.extend(
            f'nwr(around:{radius_m},{latitude},{longitude})[{clause}];'
            for clause in clauses[1:-1].split("][")
        )
    query = "[out:json][timeout:20];(" + "".join(selectors) + ");out center;"
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=urlencode({"data": query}).encode("ascii"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "AI-Estate-Valuation/1.0 (locality amenity scores)",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"OpenStreetMap amenity lookup failed: {error}") from error

    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise RuntimeError("OpenStreetMap returned an invalid amenity response.")

    found: dict[str, list[float]] = {category: [] for category in AMENITY_CATEGORIES}
    seen_elements: set[tuple[str, int]] = set()
    for element in elements:
        tags = element.get("tags", {})
        point = element.get("center", element)
        element_lat = point.get("lat")
        element_lon = point.get("lon")
        element_key = (element.get("type", ""), element.get("id", -1))
        if (
            element_lat is None
            or element_lon is None
            or element_key in seen_elements
        ):
            continue
        seen_elements.add(element_key)
        category = _amenity_category(tags)
        if category:
            found[category].append(
                _distance_metres(
                    latitude,
                    longitude,
                    float(element_lat),
                    float(element_lon),
                )
            )

    return {
        category: {
            "count": len(distances),
            "nearest_distance_m": min(distances) if distances else None,
            "score": (
                round(max(0, 100 * (1 - min(distances) / radius_m)))
                if distances
                else 0
            ),
        }
        for category, distances in found.items()
    }


def _amenity_category(tags: dict[str, str]) -> str | None:
    amenity = tags.get("amenity", "")
    if amenity in {"school", "college", "university"}:
        return "Schools"
    if amenity in {"hospital", "clinic"}:
        return "Hospitals"
    if (
        tags.get("railway") in {"station", "tram_stop"}
        or tags.get("highway") == "bus_stop"
        or tags.get("public_transport") in {"station", "stop_position"}
    ):
        return "Public transit"
    if (
        tags.get("shop") in {"mall", "supermarket"}
        or amenity == "marketplace"
    ):
        return "Commercial centers"
    if amenity in {"restaurant", "cafe", "bar", "pub", "fast_food", "food_court"}:
        return "Dining & nightlife"
    if (
        tags.get("leisure") in {"park", "garden", "nature_reserve"}
        or tags.get("landuse") == "recreation_ground"
    ):
        return "Parks & green space"
    if (
        amenity in {"police", "fire_station"}
        or tags.get("highway") in {"footway", "pedestrian", "path"}
    ):
        return "Safety / walkability"
    return None


def _distance_metres(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    earth_radius_m = 6_371_000
    lat_delta = math.radians(latitude_b - latitude_a)
    lon_delta = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(lat_delta / 2) ** 2
        + math.cos(math.radians(latitude_a))
        * math.cos(math.radians(latitude_b))
        * math.sin(lon_delta / 2) ** 2
    )
    return 2 * earth_radius_m * math.asin(min(1, math.sqrt(haversine)))


def explain_xgboost_prediction(
    model: Any,
    property_features: pd.DataFrame,
    limit: int = 8,
) -> list[tuple[str, float]]:
    """Return per-property TreeSHAP contributions for an XGBoost prediction."""
    if not hasattr(model, "get_booster"):
        raise TypeError("Per-property explanations require an XGBoost model.")
    contributions = model.get_booster().predict(
        DMatrix(property_features, feature_names=list(property_features.columns)),
        pred_contribs=True,
    )[0]
    if len(contributions) != len(property_features.columns) + 1:
        raise RuntimeError("XGBoost returned an unexpected explanation shape.")
    named_contributions = list(
        zip(property_features.columns, contributions[:-1].tolist())
    )
    ranked = sorted(
        ((str(name), float(value)) for name, value in named_contributions),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    prioritized_names = {
        "sqft_living",
        "bedrooms",
        "house_age",
        "image_quality",
        "image_finish_score",
        "location_score",
    }
    prioritized = [item for item in ranked if item[0] in prioritized_names]
    prioritized_names_present = {item[0] for item in prioritized}
    additional = [
        item for item in ranked
        if item[0] not in prioritized_names_present
    ]
    return (prioritized + additional)[:limit]


def generate_valuation_pdf(
    *,
    property_name: str,
    predicted_value: float,
    lower_bound: float,
    upper_bound: float,
    attributes: dict[str, str],
    area_name: str,
    latitude: float,
    longitude: float,
    feature_contributions: list[tuple[str, float]],
    photos: list[dict[str, Any]],
    exchange_rate: float,
    exchange_rate_updated_at: str,
) -> bytes:
    """Build a styled INR valuation report with a map link and photo thumbnails."""
    if not math.isfinite(exchange_rate) or exchange_rate <= 0:
        raise ValueError("A valid USD-to-INR exchange rate is required for the report.")

    def pdf_rupees(amount: float) -> str:
        # ReportLab's built-in Helvetica font cannot encode the rupee glyph.
        return "INR " + format_indian_rupees(amount)[1:]

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#0B5BC2"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475467"),
        )
    )
    story: list[Any] = [
        Paragraph("AI Estate — Property Valuation Report", styles["ReportTitle"]),
        Paragraph(
            f"<b>Property:</b> {html_escape(property_name)}",
            styles["BodyText"],
        ),
        Spacer(1, 0.14 * inch),
        Paragraph(
            "<b>Executive summary.</b> The machine-learning estimate is "
            f"<b>{pdf_rupees(predicted_value * exchange_rate)}</b>, "
            "with an estimated 90% residual interval of "
            f"<b>{pdf_rupees(lower_bound * exchange_rate)}–"
            f"{pdf_rupees(upper_bound * exchange_rate)}</b>. "
            "This is an automated estimate, not an appraisal or guarantee. "
            f"Converted at 1 USD = INR {exchange_rate:,.4f} "
            f"(rate updated {html_escape(exchange_rate_updated_at)}).",
            styles["BodyText"],
        ),
        Spacer(1, 0.2 * inch),
        Paragraph("Property attributes", styles["Heading2"]),
    ]
    story.append(
        _report_table(
            [["Attribute", "Value"], *[[name, value] for name, value in attributes.items()]]
        )
    )
    map_url = f"https://www.openstreetmap.org/?mlat={latitude:.6f}&mlon={longitude:.6f}#map=16/{latitude:.6f}/{longitude:.6f}"
    story.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Map preview and location", styles["Heading2"]),
            Paragraph(
                f"<b>Locality:</b> {html_escape(area_name or 'Not available')}<br/>"
                f"<b>Coordinates:</b> {latitude:.6f}, {longitude:.6f}<br/>"
                f'<link href="{map_url}" color="#0B5BC2">Open this pin in OpenStreetMap</link>',
                styles["BodyText"],
            ),
            Spacer(1, 0.18 * inch),
            Paragraph("Model feature contributions", styles["Heading2"]),
            Paragraph(
                "Per-property XGBoost TreeSHAP contributions (positive values raise "
                "this prediction; negative values lower it). They are explanatory "
                "model effects, not causal property-value adjustments.",
                styles["SmallNote"],
            ),
            Spacer(1, 0.08 * inch),
        ]
    )
    story.append(
        _report_table(
            [["Feature", "Contribution (INR)"]]
            + [
                [name, pdf_rupees(value * exchange_rate)]
                for name, value in feature_contributions
            ]
        )
    )

    if photos:
        story.extend(
            [
                Spacer(1, 0.18 * inch),
                Paragraph("Property photo thumbnails", styles["Heading2"]),
            ]
        )
        image_cells = []
        for photo in photos[:6]:
            image = PdfImage(
                BytesIO(photo["content"]),
                width=1.55 * inch,
                height=1.05 * inch,
                kind="proportional",
            )
            image_cells.append(
                [image, Paragraph(str(photo.get("category", "Property photo")), styles["SmallNote"])]
            )
        story.append(
            Table(
                [image_cells[index:index + 3] for index in range(0, len(image_cells), 3)],
                colWidths=[2.0 * inch] * min(3, len(image_cells)),
                hAlign="LEFT",
            )
        )

    story.extend(
        [
            Spacer(1, 0.2 * inch),
            Paragraph(
                "Valuation range is derived from the model's held-out validation "
                "residual error. Rental, mortgage, and amenity results are estimates "
                "and may not reflect actual market terms or complete OSM coverage.",
                styles["SmallNote"],
            ),
        ]
    )
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="AI Estate Property Valuation",
    )
    document.build(story)
    return buffer.getvalue()


def generate_valuation_summary_pdf(
    *,
    property_name: str,
    predicted_value: float,
    lower_bound: float,
    upper_bound: float,
    attributes: dict[str, str],
    latitude: float,
    longitude: float,
    exchange_rate: float,
    inspection_timestamp: str,
    photo_quality_summary: str = "Not available",
    location_score: str = "Not available",
) -> bytes:
    """Build a compact, single-page INR and USD valuation summary."""
    if not math.isfinite(exchange_rate) or exchange_rate <= 0:
        raise ValueError("A valid USD-to-INR exchange rate is required for the report.")

    buffer = BytesIO()
    page = pdf_canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    margin = 0.65 * inch
    content_width = page_width - 2 * margin

    page.setTitle("AI Estate Valuation Report")
    page.setFillColor(colors.HexColor("#0B1F3A"))
    page.rect(0, page_height - 82, page_width, 82, fill=1, stroke=0)
    page.setFillColor(colors.white)
    page.setFont("Helvetica-Bold", 19)
    page.drawString(margin, page_height - 42, "AI Estate Valuation Report")
    page.setFont("Helvetica", 9)
    page.drawString(
        margin,
        page_height - 60,
        f"Inspection / valuation time: {inspection_timestamp}",
    )

    y = page_height - 112
    page.setFillColor(colors.HexColor("#101828"))
    page.setFont("Helvetica-Bold", 13)
    property_lines = simpleSplit(
        property_name,
        "Helvetica-Bold",
        13,
        content_width,
    )
    for line in property_lines[:2]:
        page.drawString(margin, y, line)
        y -= 17

    y -= 10
    page.setFont("Helvetica-Bold", 10)
    page.drawString(margin, y, "PROPERTY DETAILS")
    y -= 16
    detail_fields = [
        ("Property type", attributes.get("Property type", "Home")),
        ("Living area", attributes.get("Living area (sq.ft)", "Not available")),
        ("Bedrooms", attributes.get("Bedrooms", "Not available")),
        ("Bathrooms", attributes.get("Bathrooms", "Not available")),
        ("Age", attributes.get("Age (years)", "Not available")),
        ("Locality", attributes.get("Locality", "Not available")),
        ("Latitude", f"{latitude:.6f}"),
        ("Longitude", f"{longitude:.6f}"),
        ("Photo clarity", photo_quality_summary),
        ("OSM / locality score", location_score),
    ]
    column_gap = 24
    column_width = (content_width - column_gap) / 2
    row_height = 35
    for row_index in range((len(detail_fields) + 1) // 2):
        for column_index in range(2):
            field_index = row_index * 2 + column_index
            if field_index >= len(detail_fields):
                continue
            label, value = detail_fields[field_index]
            x = margin + column_index * (column_width + column_gap)
            cell_y = y - row_index * row_height
            page.setFillColor(
                colors.HexColor("#F3F7FC")
                if row_index % 2 == 0
                else colors.white
            )
            page.roundRect(
                x,
                cell_y - 25,
                column_width,
                31,
                4,
                fill=1,
                stroke=0,
            )
            page.setFillColor(colors.HexColor("#475467"))
            page.setFont("Helvetica", 7)
            page.drawString(x + 8, cell_y - 5, label)
            page.setFillColor(colors.HexColor("#101828"))
            page.setFont("Helvetica-Bold", 9)
            value_lines = simpleSplit(
                str(value),
                "Helvetica-Bold",
                9,
                column_width - 16,
            )
            page.drawString(
                x + 8,
                cell_y - 18,
                value_lines[0] if value_lines else "",
            )

    y -= ((len(detail_fields) + 1) // 2) * row_height + 18
    card_height = 92
    page.setFillColor(colors.HexColor("#EAF2FF"))
    page.roundRect(margin, y - card_height, content_width, card_height, 8, fill=1, stroke=0)
    page.setFillColor(colors.HexColor("#0B1F3A"))
    page.setFont("Helvetica-Bold", 9)
    page.drawString(margin + 14, y - 20, "ESTIMATED VALUE")
    page.drawString(margin + content_width / 2 + 8, y - 20, "90% CONFIDENCE RANGE")
    page.setFont("Helvetica-Bold", 14)
    page.drawString(
        margin + 14,
        y - 42,
        f"INR {format_indian_rupees(predicted_value * exchange_rate)[1:]}",
    )
    page.setFont("Helvetica", 9)
    page.drawString(margin + 14, y - 59, f"USD ${predicted_value:,.2f}")
    page.drawString(
        margin + content_width / 2 + 8,
        y - 42,
        f"INR {format_indian_rupees(lower_bound * exchange_rate)[1:]} – "
        f"INR {format_indian_rupees(upper_bound * exchange_rate)[1:]}",
    )
    page.drawString(
        margin + content_width / 2 + 8,
        y - 59,
        f"USD ${lower_bound:,.2f} – ${upper_bound:,.2f}",
    )
    page.setFillColor(colors.HexColor("#475467"))
    page.setFont("Helvetica", 8)
    page.drawString(
        margin + 14,
        y - 78,
        f"Exchange rate: 1 USD = INR {exchange_rate:,.4f}",
    )

    footer_y = margin + 40
    page.setFillColor(colors.HexColor("#475467"))
    page.setFont("Helvetica", 8)
    page.drawString(
        margin,
        footer_y + 18,
        "Coordinates: "
        f"{latitude:.6f}, {longitude:.6f} · "
        "Automated estimate; not a formal appraisal or guarantee.",
    )
    map_url = (
        "https://www.openstreetmap.org/?mlat="
        f"{latitude:.6f}&mlon={longitude:.6f}"
        f"#map=16/{latitude:.6f}/{longitude:.6f}"
    )
    page.setFillColor(colors.HexColor("#0B5BC2"))
    page.drawString(margin, footer_y, "Open location in OpenStreetMap")
    page.linkURL(
        map_url,
        (margin, footer_y - 3, margin + 150, footer_y + 10),
        relative=0,
    )
    page.save()
    return buffer.getvalue()


def _report_table(rows: list[list[str]]) -> Table:
    table = Table(rows, colWidths=[3.5 * inch, 2.6 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#101828")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table
