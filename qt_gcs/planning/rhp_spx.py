"""RHP-SPX hybrid planner: RHP framework with SPX cell-reorder action space.

At each 25 s epoch the particle-filter belief is scored per grid cell and the
remaining unvisited cells are reordered so the highest-probability cell is
visited next.  Within each cell a boustrophedon ``local_sweep()`` pattern is
flown.  All cells are eventually visited (coverage guarantee).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor, hypot, pi
from typing import Any

import numpy as np

from cpp_search.core.models import MissionConfig, PathSegment, Point2D, SensorSpec
from cpp_search.planning.routes import local_sweep

from .geometry import LocalFrame, LocalPoint
from .rhp_fe_pf_pw_arc import IsotropicTargetParticleFilter, RHPPlanningDecision

MODEL_NAME = "RHP-SPX"


@dataclass(frozen=True, slots=True)
class _CellGrid:
    center_x: float
    center_y: float
    half_side_m: float
    width: int
    height: int

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def cell_width_m(self) -> float:
        return 2.0 * self.half_side_m / self.width

    @property
    def cell_height_m(self) -> float:
        return 2.0 * self.half_side_m / self.height

    def center_of(self, cell: int) -> Point2D:
        x = cell % self.width
        y = cell // self.width
        return Point2D(
            self.center_x - self.half_side_m + (x + 0.5) * self.cell_width_m,
            self.center_y - self.half_side_m + (y + 0.5) * self.cell_height_m,
        )

    def cell_of_point(self, p: Point2D) -> int:
        x = int((p.x - self.center_x + self.half_side_m) // self.cell_width_m)
        y = int((p.y - self.center_y + self.half_side_m) // self.cell_height_m)
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        return -1

    def cell_indices_vectorized(
        self, east: np.ndarray, north: np.ndarray
    ) -> np.ndarray:
        cw = self.cell_width_m
        ch = self.cell_height_m
        x = np.floor((east - self.center_x + self.half_side_m) / cw).astype(int)
        y = np.floor((north - self.center_y + self.half_side_m) / ch).astype(int)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        indices = np.where(valid, y * self.width + x, -1)
        return indices


class RHPSPXPlanner:
    """Receding-horizon planner that reorders SPX cell visits by PF belief."""

    def __init__(
        self,
        frame: LocalFrame,
        search_center: LocalPoint,
        *,
        search_radius_m: float,
        grid_width: int = 6,
        grid_height: int = 6,
        initial_assignments: tuple[tuple[int, ...], ...],
        mission_config: MissionConfig,
        sensor_spec: SensorSpec,
        decision_interval_s: float = 25.0,
        search_altitude_m: float = 600.0,
        swarm_coordination: bool = False,
        cell_sweep_lanes: int | None = None,
        complete_cell_sweep: bool = False,
        no_fly_cells: tuple[int, ...] = (),
    ) -> None:
        self.frame = frame
        self.search_center = search_center
        self.search_radius_m = float(search_radius_m)
        self.decision_interval_s = float(decision_interval_s)
        self.search_altitude_m = float(search_altitude_m)
        self.swarm_coordination = swarm_coordination
        # When True, a vehicle flies its whole per-cell boustrophedon to
        # completion before the planner hops it (single-UAV "full lawnmower per
        # cell" demo).  When False (default), a short in-cell sweep is enough and
        # the vehicle hops early for fast multi-UAV cell-to-cell coverage.
        self._complete_cell_sweep = bool(complete_cell_sweep)
        # Cells the vehicle may NOT fly through (terrain no-fly). The MILP already
        # excludes them from the baked cell assignments, so the queue never sweeps
        # one; this set additionally lets the live route DETOUR a transit leg that
        # would otherwise cut across a no-fly cell.
        self._no_fly_cells = frozenset(int(c) for c in no_fly_cells)
        self.mission_config = mission_config
        self.sensor_spec = sensor_spec

        self.grid = _CellGrid(
            center_x=search_center.east_m,
            center_y=search_center.north_m,
            half_side_m=search_radius_m,
            width=grid_width,
            height=grid_height,
        )

        self._initial_assignments = initial_assignments
        self._searcher_count = len(initial_assignments)

        self._visited: dict[int, set[int]] = {}
        self._queues: dict[int, list[int]] = {}
        self._last_decision_s: float = -1e9
        self._cached_candidates: dict[int, dict[str, Any]] = {}

        # Dense-dwell state: a vehicle sweeps one cell with a full boustrophedon
        # lawnmower before advancing to the next belief-priority cell.
        self._active_cell: dict[int, int] = {}
        self._route_end: dict[int, Point2D] = {}
        self._dwell: dict[int, int] = {}
        self._arrived: dict[int, bool] = {}

        # Per-cell sweep budget.  Default = FULL-CELL coverage: enough lanes at
        # the sensor track spacing to span the cell, each one cell long
        # (ceil(cell / track_spacing) * cell) -> the box fills the cell.  A
        # single UAV then dwells in one cell a long time; a mission that wants
        # visible CELL-TO-CELL hopping (e.g. "miss in cell 1, find in cell 2")
        # passes a smaller ``cell_sweep_lanes`` so the box fits inside the cell
        # and the vehicle completes + hops quickly.
        sweep_w = max(1.0, sensor_spec.effective_sweep_width_m)
        track_spacing = max(1.0, sensor_spec.track_spacing_m)
        cell_span = max(self.grid.cell_width_m, self.grid.cell_height_m)
        epoch_reach_m = max(
            1.0, mission_config.transit_speed_mps * self.decision_interval_s
        )
        if cell_sweep_lanes is not None and int(cell_sweep_lanes) > 0:
            lanes_to_cover = int(cell_sweep_lanes)
        else:
            lanes_to_cover = max(1, ceil(cell_span / track_spacing))
        self._cell_sweep_budget_m = float(lanes_to_cover * cell_span)
        # Advance once the vehicle is within this distance of the sweep's final
        # waypoint (the bounded sweep is essentially complete).
        self._done_threshold_m = max(sweep_w, 0.15 * cell_span)
        # Safety cap so a stalled vehicle still advances after a couple epochs.
        self._max_dwell_epochs = int(
            ceil(self._cell_sweep_budget_m / epoch_reach_m)
        ) + 1

        self._init_queues()

    def _init_queues(self) -> None:
        for searcher_idx in range(self._searcher_count):
            vehicle_id = searcher_idx + 1
            self._visited[vehicle_id] = set()
            seen: set[int] = set()
            queue: list[int] = []
            for cell in self._initial_assignments[searcher_idx]:
                if cell not in seen and 0 <= cell < self.grid.cell_count:
                    queue.append(cell)
                    seen.add(cell)
            self._queues[vehicle_id] = queue

    def reset(self) -> None:
        self._visited.clear()
        self._queues.clear()
        self._last_decision_s = -1e9
        self._cached_candidates.clear()
        self._active_cell.clear()
        self._route_end.clear()
        self._dwell.clear()
        self._arrived.clear()
        self._init_queues()

    @property
    def encounter_sample_count(self) -> int:
        return 0

    @property
    def radial_shortlist_count(self) -> int:
        return 0

    def evaluate(
        self,
        *,
        elapsed_s: float,
        revision: int,
        track_id: int,
        belief: IsotropicTargetParticleFilter,
        vehicles: list[dict[str, Any]],
    ) -> RHPPlanningDecision:
        eligible = [
            v for v in vehicles
            if bool(v.get("mission_launched"))
            and not bool(v.get("emergency_mode"))
            and str(v.get("flight_phase")) == "ROUTE"
            and bool(v.get("search_started", False))
        ]
        if not eligible:
            return RHPPlanningDecision({}, {})

        decision_due = (
            any(int(v.get("runtime_route_revision", 0)) == 0 for v in eligible)
            or elapsed_s - self._last_decision_s >= self.decision_interval_s
        )

        if not decision_due:
            return RHPPlanningDecision(self._cached_candidates, {})

        self._last_decision_s = elapsed_s
        cell_scores = self._score_cells(belief)
        self._reorder_queues(cell_scores)
        assigned = self._assign_next_cells(eligible)

        route_updates: dict[int, tuple[dict[str, Any], ...]] = {}
        candidates: dict[int, dict[str, Any]] = {}

        for vehicle in eligible:
            vid = int(vehicle["vehicle_id"])
            cell = assigned.get(vid)
            if cell is None:
                # Still sweeping its active cell (dense dwell): keep the prior
                # candidate and let the already-committed route keep executing.
                active = self._active_cell.get(vid)
                if active is not None and vid in self._cached_candidates:
                    candidates[vid] = self._cached_candidates[vid]
                continue

            waypoints = self._generate_cell_route(vid, cell, vehicle)
            if waypoints:
                route_updates[vid] = waypoints
                cell_center = self.grid.center_of(cell)
                lat, lon = self.frame.to_geographic(
                    LocalPoint(cell_center.x, cell_center.y)
                )
                candidates[vid] = {
                    "cell_index": cell,
                    "cell_probability": float(cell_scores[cell])
                    if cell < len(cell_scores) else 0.0,
                    "cell_center_latitude": lat,
                    "cell_center_longitude": lon,
                    "waypoint_count": len(waypoints),
                }

        self._cached_candidates = candidates
        return RHPPlanningDecision(candidates, route_updates)

    def _score_cells(self, belief: IsotropicTargetParticleFilter) -> np.ndarray:
        scores = np.zeros(self.grid.cell_count, dtype=float)
        active = ~belief.escaped
        if not np.any(active):
            return scores
        east = belief.east_m[active]
        north = belief.north_m[active]
        weights = belief.weights[active]
        indices = self.grid.cell_indices_vectorized(east, north)
        valid = indices >= 0
        if np.any(valid):
            np.add.at(scores, indices[valid], weights[valid])
        return scores

    def _reorder_queues(self, scores: np.ndarray) -> None:
        for vid, queue in self._queues.items():
            visited = self._visited.get(vid, set())
            remaining = [c for c in queue if c not in visited]
            if not remaining:
                # Queue exhausted: refill for a CONTINUOUS adaptive patrol so the
                # vehicle keeps hopping cells (re-sweeping its assigned region,
                # re-ordered by belief each epoch) instead of dwelling forever in
                # its last cell once it has visited all assigned cells.
                self._visited[vid] = set()
                seen: set[int] = set()
                idx = vid - 1
                assigned = (
                    self._initial_assignments[idx]
                    if 0 <= idx < len(self._initial_assignments) else ()
                )
                remaining = []
                for c in assigned:
                    if c not in seen and 0 <= c < self.grid.cell_count:
                        remaining.append(c)
                        seen.add(c)
            remaining.sort(key=lambda c: scores[c] if c < len(scores) else 0.0,
                           reverse=True)
            self._queues[vid] = remaining

    def _cell_done(self, vid: int, vehicle: dict[str, Any]) -> bool:
        """True once the vehicle has reached its target cell and swept it briefly,
        so the planner hops it to the next cell.  Arrival-based (reliable) rather
        than distance-to-final-sweep-waypoint (which the executed route seldom
        lands on exactly, causing the vehicle to dwell until the safety cap)."""
        active = self._active_cell.get(vid)
        if active is None:
            return True
        if self._dwell.get(vid, 0) >= self._max_dwell_epochs:
            return True
        local = self.frame.to_local(
            float(vehicle["latitude"]), float(vehicle["longitude"])
        )
        current_cell = self.grid.cell_of_point(
            Point2D(local.east_m, local.north_m)
        )
        # Latch arrival: a tangential sweep box can momentarily map the vehicle
        # into an adjacent axis-aligned cell, so an instantaneous
        # current_cell==active test misses at the 25 s epoch boundary and the
        # vehicle only advances at the (slow) dwell cap.  Once it has been inside
        # the cell, treat it as arrived.
        if current_cell == active:
            self._arrived[vid] = True
        # Arrived and dwelled >= 1 epoch (a short in-cell sweep) -> advance.
        # Skipped in complete-cell-sweep mode: there the vehicle must fly the
        # whole box, so advancement waits for the route-end/​dwell-cap checks
        # below (which are sized to the full boustrophedon budget).
        if (
            not self._complete_cell_sweep
            and self._arrived.get(vid, False)
            and self._dwell.get(vid, 0) >= 1
        ):
            return True
        # Also done if within the final-sweep threshold, as before.
        end = self._route_end.get(vid)
        if end is not None and hypot(
            local.east_m - end.x, local.north_m - end.y
        ) <= self._done_threshold_m:
            return True
        return False

    def _assign_next_cells(
        self, eligible: list[dict[str, Any]]
    ) -> dict[int, int]:
        assigned_this_epoch: set[int] = set()
        result: dict[int, int] = {}

        # Phase 1: vehicles still sweeping their active cell keep it reserved so
        # no other vehicle steals it, and their dwell counter advances.
        continuing: set[int] = set()
        for vehicle in eligible:
            vid = int(vehicle["vehicle_id"])
            active = self._active_cell.get(vid)
            if active is not None and not self._cell_done(vid, vehicle):
                assigned_this_epoch.add(active)
                continuing.add(vid)
                self._dwell[vid] = self._dwell.get(vid, 0) + 1

        # Phase 2: vehicles that finished (or never had a cell) pick the next
        # highest-belief cell from their reordered queue.
        for vehicle in eligible:
            vid = int(vehicle["vehicle_id"])
            if vid in continuing:
                continue
            queue = self._queues.get(vid, [])
            current = self._active_cell.get(vid)
            chosen = None
            # Prefer a DIFFERENT cell than the one just finished so the vehicle
            # keeps progressing cell-to-cell instead of re-sweeping in place
            # (which reads as dwelling/spinning even though it is belief-optimal).
            for cell in queue:
                if cell not in assigned_this_epoch and cell != current:
                    chosen = cell
                    break
            if chosen is None:  # fallback: only the current cell remains free
                for cell in queue:
                    if cell not in assigned_this_epoch:
                        chosen = cell
                        break
            if chosen is not None:
                assigned_this_epoch.add(chosen)
                result[vid] = chosen
                self._visited.setdefault(vid, set()).add(chosen)
                self._queues[vid] = [c for c in queue if c != chosen]
                self._active_cell[vid] = chosen
                self._dwell[vid] = 0
                self._arrived[vid] = False
        return result

    def _reorder_sweep_nearest(
        self,
        sweep_segments: list[PathSegment],
        vehicle_pos: Point2D,
    ) -> list[PathSegment]:
        """Re-anchor a ``local_sweep`` boustrophedon to enter at the lane nearest
        the vehicle, dropping the vendored routine's centre->outermost-lane hop.

        ``local_sweep`` builds the path from the cell centre and lays the first
        lane at the outer edge, so the vehicle crosses the cell before it starts
        the zigzag.  This keeps the same parallel lanes but walks them starting
        from whichever edge is closest, entering each lane at its nearer end, so
        the vehicle begins sweeping immediately with no crossing.
        """
        lanes = [s for s in sweep_segments if s.sensor_on]
        if not lanes:
            return sweep_segments

        def _mid(seg: PathSegment) -> Point2D:
            return Point2D(
                (seg.start.x + seg.end.x) / 2.0,
                (seg.start.y + seg.end.y) / 2.0,
            )

        # Lanes arrive in perpendicular-offset order; walk them from the end
        # closest to the vehicle so coverage stays edge-to-edge.
        if vehicle_pos.distance_to(_mid(lanes[0])) > vehicle_pos.distance_to(
            _mid(lanes[-1])
        ):
            lanes = list(reversed(lanes))

        out: list[PathSegment] = []
        current = vehicle_pos
        for lane in lanes:
            a, b = lane.start, lane.end
            if current.distance_to(a) <= current.distance_to(b):
                near, far = a, b
            else:
                near, far = b, a
            if current.distance_to(near) > 1e-9:
                out.append(PathSegment(current, near, False))
            out.append(replace(lane, start=near, end=far))
            current = far
        return out

    def _segment_hits_no_fly(self, a: Point2D, b: Point2D) -> bool:
        """True if the straight leg a->b passes through any no-fly cell."""
        if not self._no_fly_cells:
            return False
        span = max(self.grid.cell_width_m, self.grid.cell_height_m)
        steps = max(1, int(a.distance_to(b) / (0.4 * span)))
        for i in range(steps + 1):
            t = i / steps
            p = Point2D(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
            if self.grid.cell_of_point(p) in self._no_fly_cells:
                return True
        return False

    def _detour_waypoint(self, a: Point2D, b: Point2D) -> Point2D | None:
        """A single perpendicular-offset midpoint that makes a->w->b both clear."""
        dx, dy = b.x - a.x, b.y - a.y
        length = hypot(dx, dy)
        if length < 1e-6:
            return None
        px, py = -dy / length, dx / length  # unit perpendicular
        mx, my = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
        span = max(self.grid.cell_width_m, self.grid.cell_height_m)
        for k in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            for sign in (1.0, -1.0):
                w = Point2D(mx + px * span * k * sign, my + py * span * k * sign)
                if not self._segment_hits_no_fly(a, w) and not self._segment_hits_no_fly(w, b):
                    return w
        return None

    def _detour_no_fly(self, segments: list[PathSegment]) -> list[PathSegment]:
        """Reroute any leg that cuts across a no-fly cell around it (sensor-off
        detour hop). Lanes lie inside a flyable cell so they seldom trigger."""
        if not self._no_fly_cells or not segments:
            return segments
        out: list[PathSegment] = []
        for seg in segments:
            if not self._segment_hits_no_fly(seg.start, seg.end):
                out.append(seg)
                continue
            w = self._detour_waypoint(seg.start, seg.end)
            if w is None:
                out.append(seg)  # no clear detour found; keep (rare, heavy no-fly)
            else:
                out.append(PathSegment(seg.start, w, False))
                out.append(replace(seg, start=w))
        return out

    def _generate_cell_route(
        self,
        vehicle_id: int,
        cell_index: int,
        vehicle: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        cell_center = self.grid.center_of(cell_index)

        local = self.frame.to_local(
            float(vehicle["latitude"]), float(vehicle["longitude"])
        )
        vehicle_pos = Point2D(local.east_m, local.north_m)

        transit_distance = vehicle_pos.distance_to(cell_center)

        # Dense dwell: generate a full boustrophedon that covers the whole cell
        # (track spacing == sweep width).  The vehicle flies this multi-epoch
        # route to completion before the planner advances it to the next cell.
        budget_m = self._cell_sweep_budget_m

        phase_rad = (vehicle_id + cell_index) * 2.0 * pi / max(self._searcher_count, 1)

        sweep_segments, sweep_end = local_sweep(
            cell_center, budget_m, self.mission_config, self.sensor_spec,
            phase_rad=phase_rad,
        )
        # Enter at the lane NEAREST the vehicle instead of the vendored routine's
        # "start at cell centre -> hop to the outermost lane" ordering.  That hop
        # made the vehicle cross the cell before beginning the zigzag; re-anchor
        # the boustrophedon so it starts beside the vehicle and sweeps across.
        # (Its leading connector also replaces the transit-to-centre below.)
        sweep_segments = self._reorder_sweep_nearest(sweep_segments, vehicle_pos)
        # Route any transit leg around no-fly terrain instead of cutting through it.
        sweep_segments = self._detour_no_fly(sweep_segments)
        if sweep_segments:
            sweep_end = sweep_segments[-1].end
        # Record the final scan waypoint so _cell_done can detect completion.
        self._route_end[vehicle_id] = (
            Point2D(sweep_end.x, sweep_end.y)
            if sweep_segments else cell_center
        )

        payload: list[dict[str, Any]] = []
        seq = 1

        for segment in sweep_segments:
            for pt in (segment.start, segment.end):
                lat, lon = self.frame.to_geographic(LocalPoint(pt.x, pt.y))
                phase_code = "S" if segment.sensor_on else "T"
                payload.append({
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_m": self.search_altitude_m,
                    "code": f"RHP-{phase_code}{seq:02d}",
                    "label": (
                        f"RHP SCAN {seq}" if segment.sensor_on
                        else f"RHP TRANSIT {seq}"
                    ),
                    "point_type": (
                        "RHP_SCAN_WAYPOINT" if segment.sensor_on
                        else "RHP_TRANSIT_WAYPOINT"
                    ),
                    "sequence": seq,
                })
                seq += 1

        return tuple(payload)
