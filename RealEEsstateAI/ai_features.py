"""Optional AI listing generation and renovation planning helpers."""

from __future__ import annotations

import json
import math
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def generate_property_listing(
    property_details: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate factual listing copy through the configured OpenAI API."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the app environment to use listing generation."
        )

    safe_details = {
        str(name): value
        for name, value in property_details.items()
        if value is not None and value != ""
    }
    if not safe_details:
        raise ValueError("Property details are required to generate listing copy.")

    request_body = {
        "model": model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write polished real-estate listing copy using only supplied "
                    "facts. Do not invent amenities, legal status, condition, views, "
                    "transport access, or guarantees. Return JSON with exactly the "
                    "keys headline (string), key_features (array of 3-6 strings), "
                    "and description (string)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(safe_details, ensure_ascii=False),
            },
        ],
    }
    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Listing generation API returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not generate property listing: {error}") from error

    try:
        content = payload["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Listing generation returned an invalid response format."
        ) from error

    headline = result.get("headline")
    features = result.get("key_features")
    description = result.get("description")
    if (
        not isinstance(headline, str)
        or not headline.strip()
        or not isinstance(features, list)
        or not features
        or not all(isinstance(item, str) and item.strip() for item in features)
        or not isinstance(description, str)
        or not description.strip()
    ):
        raise RuntimeError("Listing generation returned incomplete listing content.")
    return {
        "headline": headline.strip(),
        "key_features": [item.strip() for item in features],
        "description": description.strip(),
    }


def renovation_cost_range(
    upgrade: str,
    living_area_sqft: float,
) -> tuple[float, float]:
    """Return an indicative India-market renovation cost range in INR."""
    if not math.isfinite(living_area_sqft) or living_area_sqft <= 0:
        raise ValueError("Living area must be a positive finite number.")
    fixed_costs = {
        "Add one bathroom": (250_000.0, 650_000.0),
        "Kitchen renovation": (350_000.0, 1_200_000.0),
        "Solar panels": (250_000.0, 700_000.0),
    }
    if upgrade == "Modern flooring":
        return living_area_sqft * 120.0, living_area_sqft * 350.0
    if upgrade not in fixed_costs:
        raise ValueError(f"Unsupported renovation option: {upgrade}")
    return fixed_costs[upgrade]
