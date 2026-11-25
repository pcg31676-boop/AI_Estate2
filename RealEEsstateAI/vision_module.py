"""Pretrained image embeddings and lightweight property-photo insights."""

from collections import Counter
from math import ceil
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


EMBEDDING_SIZE = 512
EMBEDDING_COLUMNS = [f"image_embedding_{index:03d}" for index in range(EMBEDDING_SIZE)]


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights).cpu().eval()
        return model, weights.transforms(), weights.meta["categories"]
    except Exception:
        return None, None, None


def _open_image(image):
    if isinstance(image, (str, Path)):
        source = Image.open(image)
    else:
        source = Image.open(BytesIO(image))
    return ImageOps.exif_transpose(source).convert("RGB")


def _quality_signals(image):
    pixels = np.asarray(image.resize((256, 256)), dtype=np.float32) / 255.0
    gray = pixels.mean(axis=2)
    brightness = float(gray.mean())
    sharpness = float(
        (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
        / 2
    )
    lighting_score = max(0.0, 1.0 - abs(brightness - 0.55) / 0.55)
    focus_score = min(1.0, sharpness / 0.12)
    quality_score = 100.0 * (0.45 * lighting_score + 0.55 * focus_score)
    contrast = float(np.mean(pixels.std(axis=(0, 1))))
    contrast_score = min(1.0, contrast / 0.25)
    finish_score = 1.0 + 4.0 * (
        0.4 * lighting_score + 0.4 * focus_score + 0.2 * contrast_score
    )
    return quality_score, brightness, sharpness, finish_score


def assess_photo_quality(image):
    """Rate photo clarity and lighting without inferring property condition."""
    photo = _open_image(image)
    quality_score, _, _, _ = _quality_signals(photo)
    return {
        "quality_score": quality_score,
        "stars": max(1, min(5, ceil(quality_score / 20))),
    }


def _room_type(labels):
    groups = {
        "Kitchen": ("kitchen", "stove", "oven", "refrigerator", "dishwasher"),
        "Bedroom": ("bedroom", "bed", "crib"),
        "Bathroom": ("bathtub", "shower", "toilet", "washbasin"),
        "Living room": ("sofa", "couch", "fireplace", "television", "coffee table"),
        "Dining room": ("dining table", "china cabinet"),
        "Open land / landscape": (
            "mountain", "meadow", "seashore", "lakeside", "valley", "desert",
            "sandbar", "cliff", "promontory", "volcano", "alp", "field",
        ),
        "Exterior": ("patio", "porch", "greenhouse", "lawn mower", "fountain", "villa"),
    }
    for label in labels:
        normalized = label.lower()
        for room, terms in groups.items():
            if any(term in normalized for term in terms):
                return room
    return "Room type unclear"


def analyze_image(image):
    """Return a ResNet embedding and human-readable signals for one photo."""
    photo = _open_image(image)
    quality, brightness, sharpness, finish_score = _quality_signals(photo)
    signals = {
        "image_quality": quality,
        "image_brightness": brightness,
        "image_sharpness": sharpness,
        "image_finish_score": finish_score,
        "image_present": 1.0,
    }

    try:
        import torch

        model, preprocess, categories = _load_model()
        if model is None or preprocess is None or categories is None:
            raise RuntimeError("Vision model could not be loaded")
        tensor = preprocess(photo).unsqueeze(0)

        with torch.inference_mode():
            features = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            features = model.layer4(features)
            embedding = model.avgpool(features).flatten(1)[0]
            logits = model.fc(embedding.unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)
            indices = torch.topk(probabilities, k=3).indices.tolist()

        labels = [categories[index] for index in indices]
        signals.update(
            {column: float(value) for column, value in zip(EMBEDDING_COLUMNS, embedding.tolist())}
        )
        room = _room_type(labels)
    except Exception:
        labels = ["Visual feature extraction unavailable"]
        signals.update({column: 0.0 for column in EMBEDDING_COLUMNS})
        room = "Room type unclear"

    return {
        "features": signals,
        "room_type": room,
        "key_visual_features": labels,
        "quality_score": quality,
    }


def analyze_images(images):
    """Average features across photos while retaining per-photo insights."""
    if not images:
        raise ValueError("Upload at least one property image.")

    analyses = [analyze_image(image) for image in images]
    numeric_keys = analyses[0]["features"].keys()
    features = {
        key: float(np.mean([item["features"][key] for item in analyses]))
        for key in numeric_keys
    }
    room_counts = Counter(item["room_type"] for item in analyses)
    visual_features = list(
        dict.fromkeys(
            label
            for item in analyses
            for label in item["key_visual_features"]
        )
    )[:5]
    return {
        "features": features,
        "room_type": room_counts.most_common(1)[0][0],
        "key_visual_features": visual_features,
        "quality_score": features["image_quality"],
        "photos": analyses,
    }


def analyze_land_images(images):
    """Return photo cues for land inspection; not a boundary or soil survey."""
    if not images:
        raise ValueError("Upload at least one land inspection photo.")

    analysis = analyze_images(images)
    vegetation_fractions = []
    exposed_soil_fractions = []
    edge_fractions = []
    for source in images:
        photo = _open_image(source).resize((256, 256))
        pixels = np.asarray(photo, dtype=np.float32) / 255.0
        red, green, blue = (pixels[:, :, channel] for channel in range(3))
        vegetation_mask = (
            (green > red * 1.12)
            & (green > blue * 1.05)
            & (green > 0.18)
        )
        soil_mask = (
            (red > green * 1.04)
            & (green > blue * 1.08)
            & (red > 0.16)
        )
        gray = pixels.mean(axis=2)
        gradient = np.hypot(
            np.gradient(gray, axis=0),
            np.gradient(gray, axis=1),
        )
        vegetation_fractions.append(float(vegetation_mask.mean()))
        exposed_soil_fractions.append(float(soil_mask.mean()))
        edge_fractions.append(float((gradient > 0.12).mean()))

    analysis["land_visual_cues"] = {
        "vegetation_area_pct": 100 * float(np.mean(vegetation_fractions)),
        "earth_tone_area_pct": 100 * float(np.mean(exposed_soil_fractions)),
        "edge_density_pct": 100 * float(np.mean(edge_fractions)),
        "recognized_tags": analysis["key_visual_features"],
    }
    return analysis