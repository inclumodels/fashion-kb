"""
Handles image input from the frontend.
Accepts base64 clipboard paste or URL.
Returns image data for Gemini Vision + embedding for DB search.
"""

import base64
import re
import requests
from io import BytesIO
from PIL import Image


def from_base64(data_url: str) -> dict:
    """
    Parse a data URL (e.g. data:image/jpeg;base64,/9j/...)
    Returns { mime_type, data (bytes) }
    """
    match = re.match(r"data:(image/\w+);base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid image data URL")
    mime_type   = match.group(1)
    image_bytes = base64.b64decode(match.group(2))
    return {"mime_type": mime_type, "data": image_bytes}


def from_url(url: str) -> dict:
    """Download image from URL, return { mime_type, data (bytes) }"""
    resp = requests.get(url, timeout=15,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    return {"mime_type": content_type, "data": resp.content}


def to_pil(image_data: dict) -> Image.Image:
    """Convert image dict to PIL Image for CLIP embedding."""
    return Image.open(BytesIO(image_data["data"])).convert("RGB")


def save_temp(image_data: dict, path: str):
    """Save image bytes to a temp file path."""
    with open(path, "wb") as f:
        f.write(image_data["data"])
