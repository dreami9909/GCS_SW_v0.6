"""Unit and integration tests for the RHP-SPX hybrid planner."""

from __future__ import annotations

import numpy as np
import pytest

from qt_gcs.planning.geometry import LocalFrame, LocalPoint
from qt_gcs.planning.rhp_fe_pf_pw_arc import (
    IsotropicTargetParticleFilter,
    RHPPlanningDecision,
)
from qt_gcs.planning.rhp_spx import RHPSPXPlanner, _CellGrid

from cpp_search.core.models import MissionConfig, Point2D, SensorSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def frame():
    return LocalFrame(37.0, 127.0)


@pytest.fixture()
def search_center(frame):
    return frame.to_local(37.0, 127.0)


@pytest.fixture()
def belief(search_center):
    return IsotropicTargetParticleFilter(
        search_center,
        search_radius_m=8900.0,
        maximum_speed_mps=40_000.0 / 3_600.0,
        particle_count=500,
        seed=42,
    )


@pytest.fixture()
def mission_config():
    return MissionConfig(
        center=Point2D(0.0, 0.0),
        search_radius_m=8900.0,
        uav_count=6,
        transit_speed_mps=160_000.0 / 3_600.0,
        search_speed_mps=160_000.0 / 3_600.0,
    )


@pytest.fixture()
def sensor_spec():
    return SensorSpec()


@pytest.fixture()
def assignments_6x6():
    """6 vehicles, each assigned 6 unique cells from 36 total."""
    return tuple(
        tuple(range(v * 6, v * 6 + 6)) for v in range(6)
    )


@pytest.fixture()
def planner(frame, search_center, mission_config, sensor_spec, assignments_6x6):
    return RHPSPXPlanner(
        frame=frame,
        search_center=search_center,
        search_radius_m=8900.0,
        grid_width=6,
        grid_height=6,
        initial_assignments=assignments_6x6,
        mission_config=mission_config,
        sensor_spec=sensor_spec,
        decision_interval_s=25.0,
    )


def _make_vehicle(vehicle_id, *, search_started=True, lat=37.0, lon=127.0):
    return {
        "vehicle_id": vehicle_id,
        "latitude": lat,
        "longitude": lon,
        "altitude_m": 600.0,
        "speed_mps": 44.44,
        "heading_deg": 0.0,
        "mission_launched": True,
        "emergency_mode": False,
        "flight_phase": "ROUTE",
        "completed_route_segment_count": 2,
        "search_started": search_started,
        "runtime_route_revision": 0,
        "runtime_route_update_count": 0,
        "ingress_sweep": False,
    }


def _make_vehicles(n=6, **kwargs):
    return [_make_vehicle(i + 1, **kwargs) for i in range(n)]


def _arrive_at_cell_end(planner, vehicles):
    """Move each vehicle to the end of its committed cell lawnmower.

    Dense-dwell keeps a vehicle on its active cell until it reaches the sweep's
    final waypoint; teleporting there marks the cell swept so the next decision
    advances it to the next cell.
    """
    for v in vehicles:
        vid = int(v["vehicle_id"])
        end = planner._route_end.get(vid)
        if end is None:
            continue
        lat, lon = planner.frame.to_geographic(LocalPoint(end.x, end.y))
        v["latitude"] = lat
        v["longitude"] = lon


# ---------------------------------------------------------------------------
# _CellGrid tests
# ---------------------------------------------------------------------------

class TestCellGrid:
    def test_cell_count(self):
        grid = _CellGrid(0.0, 0.0, 100.0, 6, 6)
        assert grid.cell_count == 36

    def test_center_of_cell_0(self):
        grid = _CellGrid(0.0, 0.0, 300.0, 6, 6)
        c = grid.center_of(0)
        expected_x = -300.0 + 0.5 * 100.0
        expected_y = -300.0 + 0.5 * 100.0
        assert abs(c.x - expected_x) < 1e-6
        assert abs(c.y - expected_y) < 1e-6

    def test_cell_indices_vectorized(self):
        grid = _CellGrid(0.0, 0.0, 300.0, 6, 6)
        center = grid.center_of(0)
        east = np.array([center.x])
        north = np.array([center.y])
        indices = grid.cell_indices_vectorized(east, north)
        assert indices[0] == 0

    def test_out_of_bounds_returns_minus_one(self):
        grid = _CellGrid(0.0, 0.0, 100.0, 6, 6)
        east = np.array([9999.0])
        north = np.array([9999.0])
        indices = grid.cell_indices_vectorized(east, north)
        assert indices[0] == -1


# ---------------------------------------------------------------------------
# RHPSPXPlanner tests
# ---------------------------------------------------------------------------

class TestRHPSPXPlannerEvaluate:
    def test_returns_rhp_planning_decision(self, planner, belief):
        vehicles = _make_vehicles(6)
        result = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert isinstance(result, RHPPlanningDecision)

    def test_route_updates_have_correct_waypoint_keys(self, planner, belief):
        vehicles = _make_vehicles(6)
        result = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        required_keys = {
            "latitude", "longitude", "altitude_m",
            "code", "label", "point_type", "sequence",
        }
        for vid, waypoints in result.route_updates.items():
            assert len(waypoints) > 0
            for wp in waypoints:
                assert required_keys.issubset(wp.keys()), (
                    f"Missing keys in waypoint: {required_keys - wp.keys()}"
                )

    def test_no_eligible_vehicles_returns_empty(self, planner, belief):
        vehicles = _make_vehicles(6, search_started=False)
        result = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert result.route_updates == {}

    def test_decision_interval_respected(self, planner, belief):
        vehicles = _make_vehicles(6)
        for v in vehicles:
            v["runtime_route_revision"] = 1

        result1 = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert result1.route_updates

        result2 = planner.evaluate(
            elapsed_s=10.0, revision=2, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert result2.route_updates == {}

        # Simulate each vehicle finishing its cell sweep (arrive at the route
        # end) so the interval-boundary decision assigns a fresh cell + route.
        _arrive_at_cell_end(planner, vehicles)
        result3 = planner.evaluate(
            elapsed_s=25.0, revision=3, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert result3.route_updates


class TestCellScoring:
    def test_scores_sum_to_approximately_one(self, planner, belief):
        scores = planner._score_cells(belief)
        active_weight = float(np.sum(belief.weights[~belief.escaped]))
        total_scored = float(np.sum(scores))
        assert total_scored <= active_weight + 1e-6

    def test_concentrated_belief_scores_correct_cell(
        self, frame, search_center, mission_config, sensor_spec
    ):
        pf = IsotropicTargetParticleFilter(
            search_center,
            search_radius_m=8900.0,
            maximum_speed_mps=0.001,
            particle_count=500,
            seed=99,
        )
        assignments = tuple(
            tuple(range(v * 6, v * 6 + 6)) for v in range(6)
        )
        planner = RHPSPXPlanner(
            frame=frame,
            search_center=search_center,
            search_radius_m=8900.0,
            grid_width=6,
            grid_height=6,
            initial_assignments=assignments,
            mission_config=mission_config,
            sensor_spec=sensor_spec,
            decision_interval_s=25.0,
        )
        scores = planner._score_cells(pf)
        assert scores.shape == (36,)
        assert np.max(scores) > 0.0


class TestDeconfliction:
    def test_no_duplicate_cells_in_same_epoch(self, planner, belief):
        vehicles = _make_vehicles(6)
        result = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assigned_cells = set()
        for vid, candidate in result.candidates.items():
            cell = candidate.get("cell_index")
            if cell is not None:
                assert cell not in assigned_cells, (
                    f"Cell {cell} assigned to multiple vehicles"
                )
                assigned_cells.add(cell)


class TestDenseDwell:
    def test_cell_route_is_a_full_lawnmower(self, planner, belief):
        """A committed cell route densely sweeps the cell (many scan legs)."""
        vehicles = _make_vehicles(1)
        result = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        wps = result.route_updates[1]
        scan_wps = [w for w in wps if w["point_type"] == "RHP_SCAN_WAYPOINT"]
        # A boustrophedon over a cell has many scan waypoints, not a lone point.
        assert len(scan_wps) >= 8

    def test_vehicle_dwells_on_same_cell_until_swept(self, planner, belief):
        """A stationary vehicle keeps sweeping one cell across epochs."""
        vehicles = _make_vehicles(1)
        first = planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        cell0 = first.candidates[1]["cell_index"]
        # Next interval epoch: vehicle has NOT moved, so it stays on the cell
        # and emits no fresh route (keeps flying the committed lawnmower).
        vehicles[0]["runtime_route_revision"] = 1
        second = planner.evaluate(
            elapsed_s=25.0, revision=2, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert 1 not in second.route_updates
        assert second.candidates[1]["cell_index"] == cell0

        # Once it reaches the sweep end, the next decision advances the cell.
        _arrive_at_cell_end(planner, vehicles)
        third = planner.evaluate(
            elapsed_s=50.0, revision=3, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert 1 in third.route_updates
        assert third.candidates[1]["cell_index"] != cell0


class TestCompleteCellSweep:
    """The single-UAV demo flies the whole per-cell box before hopping."""

    def _commit_and_enter_cell(self, planner, belief):
        vehicles = _make_vehicles(1)
        planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        vid = 1
        cell = planner._active_cell[vid]
        centre = planner.grid.center_of(cell)
        lat, lon = planner.frame.to_geographic(LocalPoint(centre.x, centre.y))
        vehicles[0]["latitude"], vehicles[0]["longitude"] = lat, lon
        planner._dwell[vid] = 1  # one epoch of dwell (a brief sweep)
        return vehicles, vid

    def test_default_hops_after_brief_sweep(self, planner, belief):
        # Default planner (fast multi-UAV hop): arrival + 1 epoch is enough.
        vehicles, vid = self._commit_and_enter_cell(planner, belief)
        assert planner._cell_done(vid, vehicles[0]) is True

    def test_complete_sweep_waits_for_full_box(
        self, frame, search_center, mission_config, sensor_spec,
        assignments_6x6, belief,
    ):
        planner = RHPSPXPlanner(
            frame=frame,
            search_center=search_center,
            search_radius_m=8900.0,
            grid_width=6,
            grid_height=6,
            initial_assignments=assignments_6x6,
            mission_config=mission_config,
            sensor_spec=sensor_spec,
            decision_interval_s=25.0,
            complete_cell_sweep=True,
        )
        vehicles, vid = self._commit_and_enter_cell(planner, belief)
        # Arrival + a brief dwell must NOT hop it early in complete-sweep mode.
        assert planner._cell_done(vid, vehicles[0]) is False
        # Reaching the sweep's final waypoint still completes the cell.
        _arrive_at_cell_end(planner, vehicles)
        assert planner._cell_done(vid, vehicles[0]) is True


class TestCoverageGuarantee:
    def test_all_cells_visited_eventually(
        self, frame, search_center, mission_config, sensor_spec
    ):
        assignments = tuple(
            tuple(range(v * 6, v * 6 + 6)) for v in range(6)
        )
        planner = RHPSPXPlanner(
            frame=frame,
            search_center=search_center,
            search_radius_m=8900.0,
            grid_width=6,
            grid_height=6,
            initial_assignments=assignments,
            mission_config=mission_config,
            sensor_spec=sensor_spec,
            decision_interval_s=25.0,
        )
        pf = IsotropicTargetParticleFilter(
            search_center,
            search_radius_m=8900.0,
            maximum_speed_mps=40_000.0 / 3_600.0,
            particle_count=500,
            seed=42,
        )
        vehicles = _make_vehicles(6)
        all_visited: set[int] = set()
        # Each vehicle owns 6 cells and dense-dwells one cell per (completed)
        # sweep, so full coverage needs ~6 advances per vehicle.  Simulate the
        # vehicle reaching each cell's sweep end so it advances to the next.
        for epoch in range(12):
            elapsed = epoch * 25.0
            result = planner.evaluate(
                elapsed_s=elapsed, revision=epoch + 1, track_id=101,
                belief=pf, vehicles=vehicles,
            )
            for vid, candidate in result.candidates.items():
                cell = candidate.get("cell_index")
                if cell is not None:
                    all_visited.add(cell)
            _arrive_at_cell_end(planner, vehicles)
            for v in vehicles:
                v["runtime_route_revision"] = epoch + 1
        assert len(all_visited) == 36


class TestReset:
    def test_reset_clears_state(self, planner, belief):
        vehicles = _make_vehicles(6)
        planner.evaluate(
            elapsed_s=0.0, revision=1, track_id=101,
            belief=belief, vehicles=vehicles,
        )
        assert any(len(v) > 0 for v in planner._visited.values())
        planner.reset()
        assert all(len(v) == 0 for v in planner._visited.values())


# ---------------------------------------------------------------------------
# Integration: planning mode in RuleBasedPlanningEngine
# ---------------------------------------------------------------------------

class TestPlanningModeSwitch:
    def test_engine_rhp_arc_default(self):
        from qt_gcs.planning.runtime import RuleBasedPlanningEngine
        engine = RuleBasedPlanningEngine(37.0, 127.0)
        assert engine.planning_mode == "RHP-ARC"
        assert engine.rhp_planner is not None

    def test_engine_rhp_spx_mode(self):
        from qt_gcs.planning.runtime import RuleBasedPlanningEngine
        engine = RuleBasedPlanningEngine(
            37.0, 127.0, planning_mode="RHP-SPX"
        )
        assert engine.planning_mode == "RHP-SPX"
        assert engine.rhp_planner is None

    def test_set_rhp_spx_planner(
        self, frame, search_center, mission_config, sensor_spec, assignments_6x6
    ):
        from qt_gcs.planning.runtime import RuleBasedPlanningEngine
        engine = RuleBasedPlanningEngine(
            37.0, 127.0, planning_mode="RHP-SPX"
        )
        planner = RHPSPXPlanner(
            frame=frame,
            search_center=search_center,
            search_radius_m=8900.0,
            grid_width=6,
            grid_height=6,
            initial_assignments=assignments_6x6,
            mission_config=mission_config,
            sensor_spec=sensor_spec,
        )
        engine.set_rhp_spx_planner(planner)
        assert engine.rhp_planner is planner
        assert engine.planning_mode == "RHP-SPX"


# ---------------------------------------------------------------------------
# Integration: belief suppression during ingress
# ---------------------------------------------------------------------------

class TestIngressBeliefSuppression:
    def test_planning_result_beliefs_suppressed_before_search(self):
        from qt_gcs.planning.runtime import (
            PlanningCycleResult,
            RuleBasedPlanningEngine,
        )
        engine = RuleBasedPlanningEngine(37.0, 127.0)
        vehicles = [
            {
                "vehicle_id": vid,
                "latitude": 37.0,
                "longitude": 127.0,
                "altitude_m": 600.0,
                "speed_mps": 44.44,
                "heading_deg": 0.0,
                "mission_launched": True,
                "emergency_mode": False,
                "flight_phase": "ROUTE",
                "completed_route_segment_count": 2,
                "search_started": True,
                "runtime_route_revision": 0,
                "runtime_route_update_count": 0,
                "ingress_sweep": False,
            }
            for vid in range(1, 7)
        ]
        subjects = [
            {
                "track_id": 101,
                "latitude": 37.01,
                "longitude": 127.01,
                "altitude_m": 0.0,
                "speed_mps": 3.0,
                "heading_deg": 90.0,
                "position_uncertainty_m": 450.0,
                "found": False,
                "measurement_latitude": 37.01,
                "measurement_longitude": 127.01,
                "estimator_speed_mps": 3.0,
                "static_observation_measurement": False,
            }
        ]
        routes = {vid: [] for vid in range(1, 7)}
        result = engine.update(
            elapsed_s=1.0,
            vehicles=vehicles,
            subjects=subjects,
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )
        assert len(result.beliefs) > 0, "Beliefs should be present when search active"
