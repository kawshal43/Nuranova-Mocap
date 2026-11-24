"""Entry point for Nuranova Mocap Studio Windows desktop application."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from desktop.ui.main_window import NuranovaMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Nuranova Mocap Studio")
    app.setOrganizationName("Nuranova Solutions")
    app.setApplicationVersion("1.0.0")

    window = NuranovaMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
