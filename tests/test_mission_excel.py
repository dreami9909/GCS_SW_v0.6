from __future__ import annotations

import unittest
from pathlib import Path

from qt_gcs.mission_excel import load_mission_workbook


class MissionExcelTests(unittest.TestCase):
    def test_upload_template_imports_complete_plan(self) -> None:
        template = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "mission_excel_upload"
            / "mission_upload_template.xlsx"
        )
        imported = load_mission_workbook(template)

        self.assertEqual({"GCS", "RDR", "LC"}, set(imported.sites))
        self.assertEqual(3, len(imported.waypoints))
        self.assertEqual([1, 2, 3], [point.sequence for point in imported.waypoints])
        self.assertIsNone(imported.return_point)
        self.assertEqual(1, len(imported.zones))
        self.assertEqual("SAFE", imported.zones[0].zone_type)
        self.assertEqual(3, len(imported.zones[0].vertices))


if __name__ == "__main__":
    unittest.main()
