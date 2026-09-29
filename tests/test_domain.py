from __future__ import annotations

import unittest

from qgc_ui.domain import MissionCommand, MissionStore
from qgc_ui.tactical import TacticalState


class MissionStoreTests(unittest.TestCase):
    def test_demo_mission_has_distance_and_selection(self) -> None:
        mission = MissionStore()
        mission.seed_demo()

        self.assertEqual(5, len(mission.waypoints))
        self.assertEqual(2, mission.selected_sequence)
        self.assertGreater(mission.total_distance_m(), 2_000)

    def test_delete_renumbers_items_and_selection(self) -> None:
        mission = MissionStore()
        mission.seed_demo()
        mission.select(3)

        self.assertTrue(mission.delete_selected())
        self.assertEqual([1, 2, 3, 4], [item.sequence for item in mission.waypoints])
        self.assertEqual(3, mission.selected_sequence)

    def test_add_and_update_selected_item(self) -> None:
        mission = MissionStore()
        added = mission.add_waypoint(37.4, 127.9, MissionCommand.LOITER)

        self.assertEqual(MissionCommand.LOITER, added.command)
        self.assertTrue(
            mission.update_selected(
                latitude=37.5,
                longitude=128.0,
                altitude_m=80,
                hold_s=15,
            )
        )
        self.assertEqual(80, mission.get_selected().altitude_m)
        self.assertEqual(15, mission.get_selected().hold_s)

    def test_tactical_state_is_non_transmitting_simulation(self) -> None:
        state = TacticalState.demo(37.3422, 127.9202)
        self.assertEqual(4, len(state.subjects))
        self.assertTrue(state.launch_ready)
        self.assertEqual("SAFE", state.automatic_mode)
        self.assertFalse(state.request_simulated_launch())

        for key in state.mission_status:
            state.mission_status[key] = True
        self.assertEqual("ARM", state.automatic_mode)
        self.assertTrue(state.request_simulated_launch())
        self.assertTrue(state.approach_success)


if __name__ == "__main__":
    unittest.main()
