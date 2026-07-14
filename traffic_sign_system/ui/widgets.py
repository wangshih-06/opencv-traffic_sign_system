"""Small animated Qt widgets used by the recognition interface."""

from __future__ import annotations

from PyQt5.QtCore import (
    QEasingCurve,
    QEvent,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    pyqtProperty,
)
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QRadialGradient
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget


class FadingLabel(QLabel):
    """A label whose new text fades in without blocking the event loop."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_animation.setDuration(300)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.setEasingCurve(QEasingCurve.OutCubic)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API spelling
        self._fade_animation.stop()
        self.setWindowOpacity(0.0)
        super().setText(text)
        self._fade_animation.start()


class ConfidenceBar(QWidget):
    """An animated 0-100 confidence indicator with an in-bar percentage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._display_value = 0.0
        self._message = ""
        self.setMinimumHeight(22)
        self.setMinimumWidth(110)
        self._animation = QPropertyAnimation(self, b"displayValue", self)
        self._animation.setDuration(500)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float | int | None) -> None:  # noqa: N802 - Qt API spelling
        value = 0.0 if value is None else max(0.0, min(100.0, float(value)))
        self._message = ""
        self._value = value
        self._animation.stop()
        self._animation.setStartValue(self._display_value)
        self._animation.setEndValue(value)
        self._animation.start()

    def reset(self) -> None:
        """Reset immediately so the next recognition animates from zero."""
        self._animation.stop()
        self._value = self._display_value = 0.0
        self._message = ""
        self.update()

    def setUnavailable(self, message: str = "不支持概率输出") -> None:  # noqa: N802 - Qt API spelling
        """Show an explanatory message when a model has no probabilities."""
        self._animation.stop()
        self._value = self._display_value = 0.0
        self._message = message
        self.update()

    def _get_display_value(self) -> float:
        return self._display_value

    def _set_display_value(self, value: float) -> None:
        self._display_value = float(value)
        self.update()

    displayValue = pyqtProperty(float, _get_display_value, _set_display_value)

    def paintEvent(self, event) -> None:
        del event
        dark = QApplication.instance().property("traffic_sign_theme") == "dark"
        track = QColor("#252535" if dark else "#E7ECF2")
        border = QColor("#3A3A4C" if dark else "#D6DBE3")
        text = QColor("#E0E0E0" if dark else "#2B2F38")
        accent = QColor("#E8833A")
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = min(5.0, rect.height() / 2.0)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(border, 1))
        painter.setBrush(track)
        painter.drawRoundedRect(rect, radius, radius)

        fill_width = max(0.0, (rect.width() - 2.0) * self._display_value / 100.0)
        if fill_width:
            fill_rect = QRectF(rect.left() + 1, rect.top() + 1, fill_width, rect.height() - 2)
            path = QPainterPath()
            path.addRoundedRect(rect.adjusted(1, 1, -1, -1), max(3.0, radius - 1), max(3.0, radius - 1))
            painter.save()
            painter.setClipPath(path)
            painter.fillRect(fill_rect, accent)
            painter.restore()

        painter.setPen(text)
        painter.drawText(
            rect,
            Qt.AlignCenter,
            self._message or f"{self._display_value:.1f}%",
        )


class _SpinnerLabel(QLabel):
    """A QLabel with an animatable rotation property for LoadingOverlay."""

    def __init__(self, parent=None):
        super().__init__("◌", parent)
        self._rotation = 0.0
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: transparent; color: #E8833A; font-size: 42px;")
        font = self.font()
        font.setPointSize(42)
        self.setFont(font)

    def _get_rotation(self) -> float:
        return self._rotation

    def _set_rotation(self, rotation: float) -> None:
        self._rotation = float(rotation)
        self.update()

    rotation = pyqtProperty(float, _get_rotation, _set_rotation)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.rect().center())
        painter.rotate(self._rotation)
        painter.translate(-self.rect().center())
        painter.setPen(QColor("#E8833A"))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())


class LoadingOverlay(QWidget):
    """A parent-covering, input-blocking overlay with a rotating loader."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(15, 16, 28, 150);")
        self.spinner = _SpinnerLabel(self)
        self.spinner.resize(64, 64)
        self.spinner.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._rotation = QPropertyAnimation(self.spinner, b"rotation", self)
        self._rotation.setStartValue(0.0)
        self._rotation.setEndValue(360.0)
        self._rotation.setDuration(900)
        self._rotation.setLoopCount(-1)
        self._rotation.setEasingCurve(QEasingCurve.Linear)
        parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() in (QEvent.Resize, QEvent.Move):
            self.setGeometry(watched.rect())
            self._center_spinner()
        return super().eventFilter(watched, event)

    def show_loading(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self._center_spinner()
        self.raise_()
        self.show()
        self._rotation.start()

    def hide_loading(self) -> None:
        self._rotation.stop()
        self.hide()

    def _center_spinner(self) -> None:
        self.spinner.move((self.width() - self.spinner.width()) // 2, (self.height() - self.spinner.height()) // 2)


class RippleButton(QPushButton):
    """A button that paints an expanding radial ripple after a pointer press."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._ripple_center = QPointF()
        self._ripple_radius = 0.0
        self._ripple_opacity = 0.0
        self._radius_animation = QPropertyAnimation(self, b"rippleRadius", self)
        self._radius_animation.setDuration(450)
        self._radius_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._opacity_animation = QPropertyAnimation(self, b"rippleOpacity", self)
        self._opacity_animation.setDuration(450)
        self._opacity_animation.setStartValue(0.34)
        self._opacity_animation.setEndValue(0.0)

    def mousePressEvent(self, event) -> None:
        self._start_ripple(event.pos())
        super().mousePressEvent(event)

    def _start_ripple(self, point) -> None:
        self._ripple_center = QPointF(point)
        self._radius_animation.stop()
        self._opacity_animation.stop()
        farthest = max(
            self._ripple_center.x(), self.width() - self._ripple_center.x(),
            self._ripple_center.y(), self.height() - self._ripple_center.y(),
        )
        self._radius_animation.setStartValue(0.0)
        self._radius_animation.setEndValue(farthest * 1.45)
        self._opacity_animation.setStartValue(0.34)
        self._opacity_animation.setEndValue(0.0)
        self._radius_animation.start()
        self._opacity_animation.start()

    def _get_ripple_radius(self) -> float:
        return self._ripple_radius

    def _set_ripple_radius(self, radius: float) -> None:
        self._ripple_radius = float(radius)
        self.update()

    def _get_ripple_opacity(self) -> float:
        return self._ripple_opacity

    def _set_ripple_opacity(self, opacity: float) -> None:
        self._ripple_opacity = float(opacity)
        self.update()

    rippleRadius = pyqtProperty(float, _get_ripple_radius, _set_ripple_radius)
    rippleOpacity = pyqtProperty(float, _get_ripple_opacity, _set_ripple_opacity)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._ripple_opacity <= 0.0 or self._ripple_radius <= 0.0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 4, 4)
        painter.setClipPath(clip)
        color = QColor(255, 255, 255, int(255 * self._ripple_opacity))
        gradient = QRadialGradient(self._ripple_center, self._ripple_radius)
        gradient.setColorAt(0.0, color)
        gradient.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(self._ripple_center, self._ripple_radius, self._ripple_radius)
