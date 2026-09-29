from __future__ import annotations

import argparse

from qgc_ui.app import QGCApplication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QGroundControl-inspired Python GCS")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the UI, process one update cycle, then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = QGCApplication(start_maximized=not args.smoke_test)
    if args.smoke_test:
        app.withdraw()
        app.update_idletasks()
        assert {"Fly", "Plan", "Analyze", "Data"} <= set(app.pages)
        app.destroy()
        print("UI_SMOKE_TEST_OK")
        return 0
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
