"""Desktop entry point for the traffic-sign recognition system.

Start the GUI with::

    python -m traffic_sign_system.main
    python -m traffic_sign_system          # (via __main__.py)

The application tries to auto-load ``models/artifacts/svm_hog+hsv.joblib``
when the file exists so that the user can start classifying immediately.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from traffic_sign_system.ui.image_utils import ensure_placeholder
from traffic_sign_system.ui.main_window import MainWindow
from traffic_sign_system.ui.styles import apply_theme

logger = logging.getLogger(__name__)

# Canonical default bundle – change this string if the file is renamed.
_DEFAULT_BUNDLE = Path("models") / "artifacts" / "svm_hog+hsv.joblib"

# Encoding-safe startup for Windows consoles that may print Chinese.
_UTF8_CONFIGURED = False


def _ensure_utf8_console() -> None:
    """Reconfigure stdout/stderr for UTF-8 on Windows when needed."""
    global _UTF8_CONFIGURED
    if _UTF8_CONFIGURED:
        return
    _UTF8_CONFIGURED = True
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass


def _resolve_default_bundle(root: Path) -> Path | None:
    """Return the absolute path to the default model bundle, or *None*."""
    candidates: list[Path] = [
        root / _DEFAULT_BUNDLE,
        root.parent / _DEFAULT_BUNDLE,  # when cwd is inside traffic_sign_system/
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def main() -> int:
    _ensure_utf8_console()

    app = QApplication(sys.argv)
    app.setApplicationName("交通标志分类识别系统")
    apply_theme(app, "light")
    ensure_placeholder()

    window = MainWindow()

    # ------------------------------------------------------------------
    # Auto-load the default SVM bundle so the user doesn't have to
    # browse for it on every launch.
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parent
    default_bundle = _resolve_default_bundle(root)
    if default_bundle is not None:
        logger.info("auto-loading default model: %s", default_bundle)
        try:
            # The LoadModelWorker path is triggered via the public API so that
            # all signal/slot wiring (status bar updates, button enables, …)
            # happens automatically.
            window._try_load_model(default_bundle)
        except Exception:
            logger.exception("failed to auto-load default model; continuing anyway")
    else:
        logger.info("default model not found; launch GUI without pre-loaded model")

    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
