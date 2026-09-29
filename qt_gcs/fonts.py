from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


def load_interface_fonts(app: QApplication) -> list[str]:
    """Load bundled OFL fonts without requiring a Windows font installation."""
    font_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    families: list[str] = []
    for path in (
        font_dir / "IBMPlexSansKR-Medium.ttf",
        font_dir / "IBMPlexSansKR-SemiBold.ttf",
        font_dir / "IBMPlexMono-Regular.ttf",
        font_dir / "IBMPlexMono-SemiBold.ttf",
    ):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families.extend(QFontDatabase.applicationFontFamilies(font_id))
    app.setFont(QFont("IBM Plex Sans KR", 10))
    return families
