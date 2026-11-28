"""Entry point for Nuranova Mocap Studio Windows desktop application."""

from __future__ import annotations

import sys
import argparse

from PySide6.QtWidgets import QApplication

from desktop.ui.main_window import NuranovaMainWindow


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", metavar="REPORT_JSON")
    arguments, qt_arguments = parser.parse_known_args(sys.argv[1:])
    app = QApplication([sys.argv[0], *qt_arguments])
    app.setApplicationName("Nuranova Mocap Studio")
    app.setOrganizationName("Nuranova Solutions")
    app.setApplicationVersion("1.0.0")

    if arguments.smoke_test:
        from desktop.smoke_test import run_smoke_test
        return run_smoke_test(app, arguments.smoke_test)

    window = NuranovaMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
