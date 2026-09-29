from __future__ import annotations

import json
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def load_map_html(api_key: str) -> tuple[str, str]:
    """Return HTML and provider name without placing the key in the source tree."""
    if api_key:
        template = (ASSET_DIR / "google_fly_3d.html").read_text(encoding="utf-8")
        return template.replace("__API_KEY_JSON__", json.dumps(api_key)), "GOOGLE 3D HYBRID"
    return (
        (ASSET_DIR / "offline_map_3d.html").read_text(encoding="utf-8"),
        "OFFLINE 3D PREVIEW",
    )
