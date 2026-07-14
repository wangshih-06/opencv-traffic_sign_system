"""Embedded Matplotlib dashboard for live recognition feedback."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget


plt.style.use("seaborn-v0_8-whitegrid")
try:
    font_manager.findfont("SimHei", fallback_to_default=False)
    rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
except ValueError:
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class DashboardWidget(QWidget):
    """Three non-blocking, theme-aware plots for model predictions."""

    _MAX_HISTORY = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme = "light"
        self._confidence: list[tuple[str, float]] = []
        self._current_class: str | None = None
        self._history: list[tuple[int, int, float]] = []
        self._prediction_index = 0
        self._confusion_image: Path | None = None
        self._scheduled: set[str] = set()

        self.tabs = QTabWidget(self)
        self.confidence_figure, self.confidence_canvas = self._new_canvas()
        self.matrix_figure, self.matrix_canvas = self._new_canvas()
        self.history_figure, self.history_canvas = self._new_canvas()
        self._add_tab(self.confidence_canvas, "置信度分布")
        self._add_tab(self.matrix_canvas, "混淆矩阵")
        self._add_tab(self.history_canvas, "预测历史")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setMinimumHeight(210)
        self.set_theme("light")

    @staticmethod
    def _new_canvas() -> tuple[Figure, FigureCanvasQTAgg]:
        figure = Figure(figsize=(5, 3), constrained_layout=True)
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumSize(0, 0)
        return figure, canvas

    def _add_tab(self, canvas: FigureCanvasQTAgg, title: str) -> None:
        page = QWidget(self.tabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        self.tabs.addTab(page, title)

    def set_theme(self, theme: str) -> None:
        """Refresh figure colours after the application QSS changes."""
        self._theme = "dark" if theme == "dark" else "light"
        self._schedule("confidence", self._draw_confidence)
        self._schedule("matrix", self._draw_confusion_matrix)
        self._schedule("history", self._draw_history)

    def update_confidence(
        self, proba_dict: Mapping[str, float], current_class: str | None = None
    ) -> None:
        """Show the ten strongest probabilities and highlight the prediction."""
        valid = (
            (str(name), max(0.0, min(1.0, float(probability))))
            for name, probability in proba_dict.items()
        )
        self._confidence = sorted(valid, key=lambda entry: entry[1], reverse=True)[:10]
        self._current_class = current_class
        self._schedule("confidence", self._draw_confidence)

    def append_prediction(self, class_id: int, confidence: float | None) -> None:
        """Append one history point, retaining only the latest 100 points."""
        if confidence is None:
            return
        self._prediction_index += 1
        self._history.append(
            (self._prediction_index, int(class_id), max(0.0, min(1.0, float(confidence))))
        )
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY :]
        self._schedule("history", self._draw_history)

    def reset_history(self) -> None:
        self._history.clear()
        self._prediction_index = 0
        self._schedule("history", self._draw_history)

    def set_confusion_matrix(self, bundle_path: Path | str | None) -> None:
        """Locate ``confusion_matrix.png`` alongside the active model bundle."""
        if bundle_path is None:
            self._confusion_image = None
        else:
            bundle = Path(bundle_path)
            candidate = (bundle if bundle.is_dir() else bundle.parent) / "confusion_matrix.png"
            self._confusion_image = candidate if candidate.is_file() else None
        self._schedule("matrix", self._draw_confusion_matrix)

    @property
    def history_count(self) -> int:
        """Expose the count for lightweight integration tests."""
        return len(self._history)

    def _schedule(self, key: str, draw_callback) -> None:
        if key in self._scheduled:
            return
        self._scheduled.add(key)

        def draw() -> None:
            self._scheduled.discard(key)
            draw_callback()

        QTimer.singleShot(0, draw)

    def _colours(self) -> tuple[str, str, str, str]:
        if self._theme == "dark":
            return "#1E1E2E", "#2A2A3C", "#E0E0E0", "#3A3A4C"
        return "#FFFFFF", "#F5F7FA", "#2B2F38", "#D6DBE3"

    def _prepare_axes(self, figure: Figure):
        figure.clear()
        face, axes_face, text, _ = self._colours()
        figure.set_facecolor(face)
        axes = figure.add_subplot(111)
        axes.set_facecolor(axes_face)
        axes.tick_params(colors=text, labelsize=8)
        axes.xaxis.label.set_color(text)
        axes.yaxis.label.set_color(text)
        axes.title.set_color(text)
        for spine in axes.spines.values():
            spine.set_color(text)
        axes.grid(color="#3A3A4C" if self._theme == "dark" else "#D6DBE3", alpha=0.55)
        return axes, text

    def _draw_confidence(self) -> None:
        axes, text = self._prepare_axes(self.confidence_figure)
        if not self._confidence:
            axes.text(0.5, 0.5, "暂无概率分布", ha="center", va="center", color=text)
            axes.set_axis_off()
        else:
            names, probabilities = zip(*self._confidence)
            colours = ["#E8833A" if name == self._current_class else "#2D5F8A" for name in names]
            positions = list(range(len(names)))
            axes.barh(positions, probabilities, color=colours)
            axes.set_yticks(positions)
            axes.set_yticklabels(names, fontsize=7)
            axes.invert_yaxis()
            axes.set_xlim(0, 1)
            axes.set_xlabel("置信度")
            axes.set_title("Top-10 类别置信度")
        self.confidence_canvas.draw_idle()

    def _draw_confusion_matrix(self) -> None:
        axes, text = self._prepare_axes(self.matrix_figure)
        if self._confusion_image is None:
            axes.text(
                0.5,
                0.5,
                "未找到 confusion_matrix.png\n请在模型 bundle 同目录生成该文件",
                ha="center",
                va="center",
                color=text,
            )
            axes.set_axis_off()
        else:
            try:
                axes.imshow(mpimg.imread(str(self._confusion_image)))
                axes.set_title("混淆矩阵")
                axes.set_axis_off()
            except OSError:
                axes.text(0.5, 0.5, "混淆矩阵图片无法读取", ha="center", va="center", color=text)
                axes.set_axis_off()
        self.matrix_canvas.draw_idle()

    def _draw_history(self) -> None:
        axes, text = self._prepare_axes(self.history_figure)
        if not self._history:
            axes.text(0.5, 0.5, "暂无预测历史", ha="center", va="center", color=text)
            axes.set_axis_off()
        else:
            grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for sequence, class_id, confidence in self._history:
                grouped[class_id].append((sequence, confidence))
            palette = plt.get_cmap("tab20")
            for index, (class_id, points) in enumerate(sorted(grouped.items())):
                sequences, confidences = zip(*points)
                axes.plot(
                    sequences,
                    confidences,
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    color=palette(index % 20),
                    label=f"ID {class_id}",
                )
            axes.set_ylim(0, 1.05)
            axes.set_xlabel("预测序号")
            axes.set_ylabel("置信度")
            axes.set_title("预测历史趋势")
            if len(grouped) <= 8:
                legend = axes.legend(fontsize=7, loc="best")
                for label in legend.get_texts():
                    label.set_color(text)
        self.history_canvas.draw_idle()
