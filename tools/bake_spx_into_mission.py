"""Bake a Stone-SPX certified search plan into a mission JSON, offline.

The Qt app no longer generates SPX plans at runtime (the "인증 탐색계획 생성"
button was removed). Instead, a scenario carries its certified plan directly:
this tool runs Stone-SPX once at authoring time and writes the resulting
per-vehicle waypoints into the mission file, so the app just loads them.

The behaviour SPX plans against comes from the mission's
``planning_motion_profile`` (see qt_gcs/planning/stone_adapter.py); if absent it
defaults to the worst case, MANEUVER_HEAVY.

After baking, ``arc_search_pattern.vehicle_arc_sequences`` is removed so that
``SiteStore.load`` keeps the baked SPX routes instead of regenerating the arc
seed waypoints from the pattern. The pattern's ``center`` and planner settings
are kept (the Fly view still reads them).

Usage:
    python -B tools/bake_spx_into_mission.py <mission.json> [--out OUT.json]
        [--uav N] [--grid N] [--terrain terrain/xxx.npz]
        [--time-limit S] [--iterations N]

Requires scipy (Stone-SPX MILP master via scipy.optimize.milp / HiGHS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qt_gcs.planning.stone_adapter import (  # noqa: E402
    apply_plan_to_store,
    plan_certified_search,
)
from qt_gcs.site_store import SiteStore  # noqa: E402


def bake(
    mission_path: str | Path,
    out_path: str | Path | None = None,
    *,
    uav_count: int | None = None,
    grid_n: int = 6,
    terrain_npz: str | None = None,
    flight_msl_m: float = 1130.0,
    min_clearance_agl_m: float = 120.0,
    master_time_limit_s: float = 60.0,
    max_iterations: int = 20,
    planning_mode: str | None = None,
) -> dict:
    mission_path = Path(mission_path)
    out_path = Path(out_path) if out_path else mission_path

    store = SiteStore()
    store.load(mission_path)

    plan = plan_certified_search(
        store,
        terrain_npz=terrain_npz,
        flight_msl_m=flight_msl_m,
        min_clearance_agl_m=min_clearance_agl_m,
        uav_count=uav_count,
        grid_n=grid_n,
        master_time_limit_s=master_time_limit_s,
        max_iterations=max_iterations,
    )
    apply_plan_to_store(store, plan)

    data = store.to_dict()
    mission = data.setdefault("mission", {})

    # Stop SiteStore.load from rebuilding arc seed waypoints over the baked plan.
    arc = mission.get("arc_search_pattern")
    if isinstance(arc, dict):
        arc.pop("vehicle_arc_sequences", None)
        arc["note"] = "arc sequences removed; routes are the baked SPX plan"

    if plan.assignments:
        mission["spx_assignments"] = [list(path) for path in plan.assignments]
    if planning_mode:
        mission["planning_mode"] = planning_mode
    # Record the terrain used + flight geometry so the live RHP-SPX planner can
    # recompute no-fly cells (in its own grid) and DETOUR transits around them.
    if terrain_npz:
        mission["spx_terrain_npz"] = str(terrain_npz).replace("\\", "/")
        mission["spx_flight_msl_m"] = float(flight_msl_m)
        mission["spx_clearance_agl_m"] = float(min_clearance_agl_m)

    cert = plan.certificate
    mission["spx_certified"] = {
        "method": cert.method,
        "certified": bool(cert.certified),
        "detection_probability": round(float(cert.detection_probability), 4),
        "relative_optimality_gap": round(float(cert.relative_optimality_gap), 4),
        "iterations": int(cert.iterations),
        "grid_shape": list(cert.grid_shape),
        "no_fly_cell_count": int(cert.no_fly_cell_count),
        "runtime_s": round(float(cert.runtime_s), 2),
        "planning_motion_profile": mission.get("planning_motion_profile", "MANEUVER_HEAVY"),
        "per_target_pd": [
            [name, round(float(pd), 4)]
            for name, pd in cert.target_detection_probabilities
        ],
    }

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return mission["spx_certified"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bake Stone-SPX plan into a mission JSON")
    p.add_argument("mission")
    p.add_argument("--out", default=None)
    p.add_argument("--uav", type=int, default=None)
    p.add_argument("--grid", type=int, default=6)
    p.add_argument("--terrain", default=None)
    p.add_argument("--flight-msl", type=float, default=1130.0)
    p.add_argument("--clearance", type=float, default=120.0)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--planning-mode", default=None,
                   help="Set planning_mode in mission metadata (e.g. RHP-SPX)")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    summary = bake(
        args.mission,
        args.out,
        uav_count=args.uav,
        grid_n=args.grid,
        terrain_npz=args.terrain,
        flight_msl_m=args.flight_msl,
        min_clearance_agl_m=args.clearance,
        master_time_limit_s=args.time_limit,
        max_iterations=args.iterations,
        planning_mode=args.planning_mode,
    )
    print("SPX baked:", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
