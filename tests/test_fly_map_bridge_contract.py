"""Guards the Fly map page <-> bridge pairing.

The standalone (non-shared) Fly view drives its own web view through
``FlyMapBridge`` (``bridge.stateChanged``). Only ``offline_fly_3d.html``
speaks that contract; the online ``google_fly_3d.html`` expects the
``MapBridge`` + ``window.setFlyState`` contract that only the shared PLAN web
view drives. Previously the non-shared path loaded the online page whenever a
Google Maps key was present, wiring a bridge the page could not talk to.
"""

from __future__ import annotations

from pathlib import Path

from qt_gcs.fly_bridge import FlyMapBridge
from qt_gcs.fly_map_html import ASSET_DIR, resolve_fly_map_html
from qt_gcs.map_bridge import MapBridge

OFFLINE_PROVIDER = "OFFLINE TACTICAL PREVIEW"
ONLINE_PROVIDER = "GOOGLE 3D HYBRID"

# Members the ONLINE google_fly_3d.html invokes on the bridge that the offline
# FlyMapBridge does not provide. Their presence in the online page is exactly
# why it must never be paired with FlyMapBridge.
MAPBRIDGE_ONLY_MEMBERS = ("planChanged", "reportMapClick", "reportFeatureSelected")


def _read(asset: str) -> str:
    return (ASSET_DIR / asset).read_text(encoding="utf-8")


def test_non_shared_context_always_uses_offline_page() -> None:
    # A key must not pull the online page into the non-shared path.
    html_key, provider_key = resolve_fly_map_html("SECRET_KEY", uses_shared_map=False)
    html_nokey, provider_nokey = resolve_fly_map_html("", uses_shared_map=False)
    assert provider_key == OFFLINE_PROVIDER
    assert provider_nokey == OFFLINE_PROVIDER
    assert "bridge.stateChanged" in html_key
    assert "bridge.stateChanged" in html_nokey


def test_shared_context_still_selects_online_page_with_key() -> None:
    html, provider = resolve_fly_map_html("SECRET_KEY", uses_shared_map=True)
    assert provider == ONLINE_PROVIDER
    assert "SECRET_KEY" in html  # key injected into the online template
    html_off, provider_off = resolve_fly_map_html("", uses_shared_map=True)
    assert provider_off == OFFLINE_PROVIDER


def test_flymapbridge_matches_offline_page_contract() -> None:
    offline = _read("offline_fly_3d.html")
    # The offline page talks the FlyMapBridge dialect...
    assert "bridge.stateChanged" in offline
    for member in ("stateChanged", "requestInitialState", "reportSubjectSelected"):
        assert hasattr(FlyMapBridge, member), member
    # ...and never reaches for MapBridge-only members.
    for member in MAPBRIDGE_ONLY_MEMBERS:
        assert not hasattr(FlyMapBridge, member), member
        assert f"bridge.{member}" not in offline, member


def test_online_page_needs_mapbridge_not_flymapbridge() -> None:
    online = _read("google_fly_3d.html")
    for member in MAPBRIDGE_ONLY_MEMBERS:
        assert f"bridge.{member}" in online, member  # online page calls it
        assert hasattr(MapBridge, member), member      # MapBridge provides it
        assert not hasattr(FlyMapBridge, member), member  # FlyMapBridge cannot
