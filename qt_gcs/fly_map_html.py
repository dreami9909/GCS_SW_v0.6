from __future__ import annotations

import json
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def load_fly_map_html(api_key: str) -> tuple[str, str]:
    if api_key:
        template = (ASSET_DIR / "google_fly_3d.html").read_text(encoding="utf-8")
        return (
            template.replace("__API_KEY_JSON__", json.dumps(api_key)),
            "GOOGLE 3D HYBRID",
        )
    return (
        (ASSET_DIR / "offline_fly_3d.html").read_text(encoding="utf-8"),
        "OFFLINE TACTICAL PREVIEW",
    )


def resolve_fly_map_html(
    api_key: str,
    *,
    uses_shared_map: bool,
) -> tuple[str, str]:
    """Pick the Fly page + provider label for the current map context.

    The standalone (non-shared) Fly view owns its own web view and drives it
    through ``FlyMapBridge`` (``bridge.stateChanged``). Only
    ``offline_fly_3d.html`` implements that contract. The online
    ``google_fly_3d.html`` instead expects the ``MapBridge`` +
    ``window.setFlyState`` / ``setMapMode`` contract that only the shared PLAN
    web view drives. Loading the online page for the non-shared path therefore
    wires a bridge the page cannot talk to (``planChanged`` / ``reportMapClick``
    / ``reportFeatureSelected`` are missing on ``FlyMapBridge``), so the map
    never renders. Select the online page only for the shared context; the
    non-shared path always uses the offline preview regardless of the key.
    """

    if uses_shared_map:
        return load_fly_map_html(api_key)
    return load_fly_map_html("")


def load_search_camera_html(api_key: str) -> str:
    template = (ASSET_DIR / "search_camera_2d.html").read_text(encoding="utf-8")
    return template.replace("__API_KEY_JSON__", json.dumps(api_key))
