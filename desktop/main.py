"""Entry point for Nuranova Mocap Studio Windows desktop application."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from desktop.ui.startup import StartupCoordinator
from desktop.ui.widgets.studio_chrome import configure_studio_font


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", metavar="REPORT_JSON")
    parser.add_argument("--startup-report", metavar="REPORT_JSON")
    parser.add_argument("--settings-path", type=Path, help=argparse.SUPPRESS)
    arguments, qt_arguments = parser.parse_known_args(sys.argv[1:])
    app = QApplication([sys.argv[0], *qt_arguments])
    app.setApplicationName("Nuranova Mocap Studio")
    app.setOrganizationName("Nuranova Solutions")
    app.setApplicationVersion("1.0.0")
    configure_studio_font(app)

    isolated = tempfile.TemporaryDirectory(prefix="nuranova_smoke_") if arguments.smoke_test else nullcontext(None)
    with isolated as workdir:
        settings_path = arguments.settings_path or (Path(workdir) / "settings.json" if workdir else None)
        startup = StartupCoordinator(app, settings_path)

        def save_startup_report():
            if arguments.startup_report:
                output = Path(arguments.startup_report).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(startup.report, indent=2), encoding="utf-8")

        def on_ready(window):
            save_startup_report()
            if arguments.smoke_test:
                from desktop.smoke_test import schedule_smoke_test
                schedule_smoke_test(app, arguments.smoke_test, window, workdir, startup.report)

        def on_failure(message):
            save_startup_report()
            if arguments.smoke_test:
                output = Path(arguments.smoke_test).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"passed": False, "error": message,
                    "startup": startup.report}, indent=2), encoding="utf-8")
                startup.screen.dismiss()
                app.exit(1)

        def capture_startup(completed, title, detail):
            if arguments.startup_report and completed == 3:
                output = Path(arguments.startup_report).resolve().with_suffix(".png")
                output.parent.mkdir(parents=True, exist_ok=True)
                startup.screen.grab().save(str(output))

        startup.ready.connect(on_ready)
        startup.failed.connect(on_failure)
        startup.progress_changed.connect(capture_startup)
        QTimer.singleShot(0, startup.start)
        return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
