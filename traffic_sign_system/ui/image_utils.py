"""Qt image conversion helpers used by the desktop interface."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap


def bgr_to_qimage(img_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV image to an owning :class:`QImage`.

    BGR, BGRA, and single-channel grayscale uint8 arrays are supported.  The
    returned image owns a copy of the pixel data, so it remains valid after the
    NumPy array goes out of scope.
    """
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("图像必须是 numpy.ndarray。")
    if img_bgr.size == 0:
        raise ValueError("图像不能为空。")

    image = np.asarray(img_bgr)
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"不支持的图像数据类型：{image.dtype}")
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        gray = np.ascontiguousarray(image)
        qimage = QImage(
            gray.data,
            gray.shape[1],
            gray.shape[0],
            int(gray.strides[0]),
            QImage.Format_Grayscale8,
        )
        return qimage.copy()

    if image.ndim != 3:
        raise ValueError(f"图像维度应为 2 或 3，实际为 {image.ndim}。")

    channels = image.shape[2]
    if channels == 3:
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        )
        return qimage.copy()

    if channels == 4:
        rgba = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        qimage = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            int(rgba.strides[0]),
            QImage.Format_RGBA8888,
        )
        return qimage.copy()

    raise ValueError(f"仅支持灰度、BGR 或 BGRA 图像，实际通道数为 {channels}。")


def qimage_to_pixmap(qimg: QImage) -> QPixmap:
    """Convert a non-null :class:`QImage` to :class:`QPixmap`."""
    if not isinstance(qimg, QImage) or qimg.isNull():
        raise ValueError("QImage 不能为空。")
    return QPixmap.fromImage(qimg)


def numpy_to_pixmap(
    img_bgr: np.ndarray,
    target_size: QSize | None = None,
) -> QPixmap:
    """Convert a NumPy image to a pixmap and optionally scale it proportionally."""
    pixmap = qimage_to_pixmap(bgr_to_qimage(img_bgr))
    if target_size is None or not target_size.isValid():
        return pixmap
    return pixmap.scaled(
        target_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


# ---------------------------------------------------------------------------
# Chinese text rendering (Pillow) with ASCII / cv2.putText fallback
# ---------------------------------------------------------------------------
#
# cv2.putText only supports the Hershey vector fonts, which cover ASCII
# 32-126 and cannot render CJK glyphs at all. Pillow is used to composite
# real Chinese text onto the image; when Pillow or a CJK font file is
# unavailable, callers fall back to an ASCII-safe string drawn with
# cv2.putText instead of silently producing garbled boxes.

PLACEHOLDER_PATH = Path(__file__).resolve().parent / "resources" / "placeholder.png"

_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_pillow_available: bool | None = None
_font_path: str | None = None
_font_path_resolved: bool = False
_font_cache: dict = {}


def _has_pillow() -> bool:
    """Return whether Pillow can be imported, caching the result."""
    global _pillow_available
    if _pillow_available is None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            _pillow_available = False
        else:
            _pillow_available = True
    return _pillow_available


def _resolve_font_path() -> str | None:
    """Return the first existing CJK font file from the candidate list."""
    global _font_path, _font_path_resolved
    if not _font_path_resolved:
        _font_path_resolved = True
        for candidate in _CJK_FONT_CANDIDATES:
            if os.path.isfile(candidate):
                _font_path = candidate
                break
    return _font_path


def _resolve_cjk_font(size: int):
    """Return a cached Pillow ``FreeTypeFont`` for *size*, or ``None``."""
    if not _has_pillow():
        return None
    path = _resolve_font_path()
    if path is None:
        return None
    key = (path, size)
    if key not in _font_cache:
        from PIL import ImageFont

        try:
            _font_cache[key] = ImageFont.truetype(path, size)
        except OSError:
            _font_cache[key] = None
    return _font_cache[key]


def _render_text_lines(
    img_bgr: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font,
    color_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Left-align *lines* starting at *origin* on a copy of *img_bgr*."""
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=color_rgb)
        _, _, _, bottom = draw.textbbox((x, y), line, font=font)
        y = bottom + 6
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _fit_cjk_font(
    lines: list[str],
    max_width: int,
    max_height: int,
    preferred_size: int,
):
    """Return the largest available CJK font that keeps all lines in bounds."""
    if not _has_pillow():
        return None

    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for size in range(max(10, preferred_size), 9, -1):
        font = _resolve_cjk_font(size)
        if font is None:
            return None
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        widths = [right - left for left, _, right, _ in boxes]
        heights = [bottom - top for _, top, _, bottom in boxes]
        total_height = sum(heights) + max(0, len(lines) - 1) * max(3, size // 5)
        if max(widths, default=0) <= max_width and total_height <= max_height:
            return font
    return _resolve_cjk_font(10)


def _render_text_centered(
    img_bgr: np.ndarray,
    text: str,
    font,
    color_rgb: tuple[int, int, int] = (60, 60, 68),
) -> np.ndarray:
    """Draw *text* centered on a copy of *img_bgr*."""
    from PIL import Image, ImageDraw

    height, width = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = max(0, (width - (right - left)) // 2)
    y = max(0, (height - (bottom - top)) // 2)
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def overlay_prediction(
    img_bgr: np.ndarray,
    result: dict,
    position: str = "bottom",
) -> np.ndarray:
    """Return a copy of *img_bgr* with a translucent banner showing *result*.

    *result* is a prediction dict as returned by ``Predictor.predict``
    (``class_id``, ``class_name``, optional ``confidence``). Chinese text is
    rendered via Pillow when available; otherwise an ASCII-safe fallback is
    drawn with ``cv2.putText``.
    """
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
        raise ValueError("img_bgr 必须是形状为 (H, W, 3) 的 BGR 图像。")

    canvas = np.ascontiguousarray(img_bgr).copy()
    height, width = canvas.shape[:2]

    # GTSRB samples are often only 25-60 pixels wide. Drawing directly on
    # those source pixels makes even a 14 px font cover the whole sign. Build
    # a display-sized annotated copy first; ImageCanvas can still scale this
    # result down responsively when the window is small.
    display_scale = max(1.0, 480.0 / max(1, width), 320.0 / max(1, height))
    if display_scale > 1.0:
        width = max(1, int(round(width * display_scale)))
        height = max(1, int(round(height * display_scale)))
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_CUBIC)

    class_id = result.get("class_id")
    class_name = str(result.get("class_name", "--"))
    confidence = result.get("confidence")
    confidence_line = (
        "置信度：本模型不支持概率输出"
        if confidence is None
        else f"置信度：{float(confidence):.2%}"
    )
    lines = [f"类别 {class_id}：{class_name}", confidence_line]

    banner_height = min(height, int(np.clip(round(height * 0.22), 76, 120)))
    y0 = 0 if position == "top" else height - banner_height
    y1 = banner_height if position == "top" else height

    shaded = canvas.copy()
    cv2.rectangle(shaded, (0, y0), (width, y1), (0, 0, 0), cv2.FILLED)
    canvas = cv2.addWeighted(shaded, 0.55, canvas, 0.45, 0)

    padding_x = max(10, int(round(width * 0.025)))
    padding_y = max(7, int(round(banner_height * 0.10)))
    available_width = max(1, width - padding_x * 2)
    available_height = max(1, banner_height - padding_y * 2)
    font_size = int(np.clip(round(banner_height / 3.2), 12, 30))
    font = _fit_cjk_font(lines, available_width, available_height, font_size)
    if font is not None:
        canvas = _render_text_lines(canvas, lines, (padding_x, y0 + padding_y), font)
    else:
        ascii_lines = [
            f"Class {class_id}",
            "Confidence: N/A" if confidence is None else f"Confidence: {float(confidence):.2%}",
        ]
        scale = max(0.4, font_size / 30.0)
        thickness = max(1, int(round(scale * 2)))
        widest = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
            for line in ascii_lines
        )
        if widest > available_width:
            scale *= available_width / widest
            thickness = max(1, int(round(scale * 2)))
        text_y = y0 + padding_y + int(font_size * 0.9)
        for line in ascii_lines:
            cv2.putText(
                canvas, line, (padding_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
            text_y += int(font_size * 1.25)

    return canvas


def ensure_placeholder(
    path: Path | None = None,
    size: tuple[int, int] = (500, 300),
) -> Path:
    """Generate the empty-state placeholder PNG if it does not already exist."""
    target = Path(path) if path is not None else PLACEHOLDER_PATH
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    width, height = size
    canvas = np.full((height, width, 3), (200, 206, 216), dtype=np.uint8)
    text = "请选择图片"
    font = _resolve_cjk_font(28)
    if font is not None:
        canvas = _render_text_centered(canvas, text, font)
    else:
        ascii_text = "Select Image"
        (text_w, text_h), _ = cv2.getTextSize(
            ascii_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        x = max(0, (width - text_w) // 2)
        y = (height + text_h) // 2
        cv2.putText(
            canvas, ascii_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (60, 60, 60), 2, cv2.LINE_AA,
        )

    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("无法编码占位图 PNG 数据。")
    target.write_bytes(buffer.tobytes())
    return target
