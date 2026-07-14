"""PyQt5 main window for image, video, and camera recognition."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from traffic_sign_system.recognition.camera_recognizer import CameraRecognizer
from traffic_sign_system.recognition.scene_aware import SceneAnalyzer
from traffic_sign_system.recognition.sign_detector import SignDetector
from traffic_sign_system.recognition.video_recognizer import VideoRecognizer
from traffic_sign_system.ui.image_utils import (
    PLACEHOLDER_PATH,
    ensure_placeholder,
    numpy_to_pixmap,
    overlay_prediction,
)
from traffic_sign_system.ui.dashboard import DashboardWidget
from traffic_sign_system.ui.workers import (
    BatchPredictWorker,
    DetectWorker,
    LoadModelWorker,
    PredictWorker,
)
from traffic_sign_system.ui.styles import toggle_theme
from traffic_sign_system.ui.widgets import ConfidenceBar, FadingLabel, LoadingOverlay, RippleButton

logger = logging.getLogger(__name__)


class ImageCanvas(QLabel):
    """A QLabel that keeps its source pixmap and rescales it on resize."""

    def __init__(
        self,
        placeholder: str,
        minimum_height: int = 165,
        placeholder_pixmap: QPixmap | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._source_pixmap = None
        self._placeholder = placeholder
        self._placeholder_pixmap = (
            placeholder_pixmap
            if placeholder_pixmap is not None and not placeholder_pixmap.isNull()
            else None
        )
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, minimum_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("imageCanvas")
        self.clear_image()

    def set_image(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self.clear_image()
            return
        self._source_pixmap = QPixmap(pixmap)
        self.setText("")
        self._refresh_pixmap()

    def clear_image(self) -> None:
        if self._placeholder_pixmap is not None:
            self._source_pixmap = QPixmap(self._placeholder_pixmap)
            self.setText("")
            self._refresh_pixmap()
        else:
            self._source_pixmap = None
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        size = QSize(max(1, self.width() - 16), max(1, self.height() - 16))
        self.setPixmap(
            self._source_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


class CollapsibleSection(QWidget):
    """A titled group of controls with a clickable header that folds them away."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self.setObjectName("sectionCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._toggle_button = QToolButton(self)
        self._toggle_button.setObjectName("collapsibleHeader")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(True)
        self._toggle_button.setCursor(Qt.PointingHandCursor)
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle_button.toggled.connect(self._on_toggled)

        self._body = QWidget(self)
        self._body.setObjectName("sectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 10, 14, 14)
        self._body_layout.setSpacing(12)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle_button)
        outer.addWidget(self._body)

        self._set_arrow(True)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def _on_toggled(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._set_arrow(checked)

    def _set_arrow(self, expanded: bool) -> None:
        arrow = "▼" if expanded else "▶"
        self._toggle_button.setText(f"{arrow} {self._title}")


class MainWindow(QMainWindow):
    IMAGE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.ppm *.bmp);;所有文件 (*)"
    MODEL_FILTER = "模型 Bundle (*.joblib);;所有文件 (*)"
    VIDEO_FILTER = "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("交通标志分类识别系统")
        self.resize(1500, 940)
        self.setMinimumSize(1120, 720)

        self.predictor = None
        self.scene_analyzer = SceneAnalyzer()
        self.current_image = None
        self.current_image_path = None
        self.load_worker = None
        self.predict_worker = None
        self.detect_worker = None
        self.batch_worker = None
        self.active_recognizer = None
        self.stream_mode = None
        self.stream_paused = False
        self.last_stream_frame = None
        self.video_writer = None
        self.pending_save_path = None
        self.output_save_path = None
        self._loading_bundle_path = None
        app = QApplication.instance()
        self.current_theme = (
            app.property("traffic_sign_theme") if app is not None else None
        ) or "light"

        self.stream_timer = QTimer(self)
        self.stream_timer.setInterval(33)
        self.stream_timer.timeout.connect(self._on_tick)

        self._create_actions()
        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()
        self.loading_overlay = LoadingOverlay(self)
        self._connect_controls()
        self._set_ready_state()

    # ------------------------------------------------------------------
    # Construction: actions / menu
    # ------------------------------------------------------------------
    def _create_actions(self):
        def action(text, slot, shortcut=None, icon=None, checkable=False):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(shortcut)
            if icon is not None:
                a.setIcon(self.style().standardIcon(icon))
            if checkable:
                a.setCheckable(True)
            return a

        SP = QStyle
        self.open_image_action = action("选择图片…", self.choose_image, "Ctrl+O", SP.SP_DialogOpenButton)
        self.select_video_action = action("选择视频…", self.choose_video, "Ctrl+V", SP.SP_FileDialogDetailedView)
        self.open_camera_action = action("打开摄像头", self.open_camera, "Ctrl+K", SP.SP_ComputerIcon)
        self.batch_predict_action = action("批量识别文件夹…", self.start_batch_predict, "Ctrl+B", SP.SP_DirOpenIcon)
        self.save_video_action = action("保存结果", self.toggle_save_result, "Ctrl+S", SP.SP_DialogSaveButton)
        self.clear_action = action("清空图片", self.clear_all, None, SP.SP_DialogResetButton)
        self.exit_action = action("退出", self.close, "Ctrl+Q", SP.SP_TitleBarCloseButton)

        self.browse_model_action = action("选择模型…", self.browse_model, "Ctrl+M", SP.SP_DirOpenIcon)
        self.load_model_action = action("加载模型", self.load_model, "Ctrl+L", SP.SP_BrowserReload)
        self.predict_action = action("图片识别", self.start_prediction, "Ctrl+R", SP.SP_DialogApplyButton)
        self.scene_detect_action = action("场景检测", self.start_scene_detection, "Ctrl+D", SP.SP_FileDialogContentsView)

        self.pause_action = action("暂停", self.toggle_pause, "Space", SP.SP_MediaPause)
        self.stop_stream_action = action("停止", self.stop_stream, "Ctrl+Shift+T", SP.SP_MediaStop)
        self.toggle_theme_action = action("切换暗色主题", self._toggle_theme, "Ctrl+T")
        self.toggle_left_panel_action = action("切换左侧面板", self._toggle_left_panel, "F9", checkable=True)
        self.toggle_right_panel_action = action("切换右侧面板", self._toggle_right_panel, "F10", checkable=True)
        self.toggle_left_panel_action.setChecked(True)
        self.toggle_right_panel_action.setChecked(True)

        self.about_action = action("关于", self.show_about, None, SP.SP_MessageBoxInformation)

    def _create_menu_bar(self):
        file_menu = self.menuBar().addMenu("文件(&F)")
        file_menu.addActions(
            [
                self.open_image_action,
                self.select_video_action,
                self.open_camera_action,
                self.batch_predict_action,
                self.save_video_action,
            ]
        )
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        model_menu = self.menuBar().addMenu("模型(&M)")
        model_menu.addActions(
            [
                self.browse_model_action,
                self.load_model_action,
                self.predict_action,
                self.scene_detect_action,
            ]
        )

        view_menu = self.menuBar().addMenu("视图(&V)")
        view_menu.addActions([self.pause_action, self.stop_stream_action])
        view_menu.addSeparator()
        view_menu.addActions([self.toggle_left_panel_action, self.toggle_right_panel_action])
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_theme_action)

        self.menuBar().addMenu("帮助(&H)").addAction(self.about_action)

    # ------------------------------------------------------------------
    # Construction: central splitter (left controls / center stack / right results)
    # ------------------------------------------------------------------
    def _create_central_widget(self):
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setHandleWidth(1)
        splitter.setSizes([300, 830, 370])

        self.main_splitter = splitter
        self.setCentralWidget(splitter)

    def _build_left_panel(self):
        def decorate(button, role: str, standard_icon) -> None:
            button.setObjectName(role)
            button.setIcon(self.style().standardIcon(standard_icon))
            button.setIconSize(QSize(17, 17))
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(38)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择 *.joblib 模型文件")
        self.browse_model_button = QPushButton("浏览…")
        self.load_model_button = QPushButton("加载模型")
        decorate(self.browse_model_button, "secondaryButton", QStyle.SP_DirOpenIcon)
        decorate(self.load_model_button, "primaryButton", QStyle.SP_BrowserReload)

        model_section = CollapsibleSection("模型")
        path_row = QHBoxLayout()
        path_row.addWidget(self.model_path_edit, 1)
        path_row.addWidget(self.browse_model_button)
        model_section.body_layout().addLayout(path_row)
        model_section.body_layout().addWidget(self.load_model_button)

        self.choose_image_button = QPushButton("选择图片")
        self.predict_button = RippleButton("识别图片")
        self.scene_detect_button = QPushButton("场景检测")
        self.clear_button = RippleButton("清空图片")
        self.adaptive_enhance_checkbox = QCheckBox("自适应增强")
        self.adaptive_enhance_checkbox.setToolTip(
            "按亮度、对比度和模糊程度自动调整 CLAHE、归一化及锐化参数"
        )
        self.adaptive_enhance_checkbox.setChecked(False)
        decorate(self.choose_image_button, "secondaryButton", QStyle.SP_DialogOpenButton)
        decorate(self.predict_button, "primaryButton", QStyle.SP_DialogApplyButton)
        decorate(self.scene_detect_button, "secondaryButton", QStyle.SP_FileDialogContentsView)
        decorate(self.clear_button, "dangerButton", QStyle.SP_DialogResetButton)
        image_section = CollapsibleSection("图片")
        for widget in (
            self.choose_image_button,
            self.predict_button,
            self.scene_detect_button,
            self.clear_button,
            self.adaptive_enhance_checkbox,
        ):
            image_section.body_layout().addWidget(widget)

        self.select_video_button = QPushButton("选择视频")
        decorate(self.select_video_button, "secondaryButton", QStyle.SP_MediaPlay)
        video_section = CollapsibleSection("视频")
        video_section.body_layout().addWidget(self.select_video_button)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2"])
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(32, 256)
        self.roi_size_spin.setSingleStep(16)
        self.roi_size_spin.setValue(64)
        self.roi_size_spin.setSuffix(" px")
        self.open_camera_button = QPushButton("打开摄像头")
        decorate(self.open_camera_button, "primaryButton", QStyle.SP_ComputerIcon)
        camera_section = CollapsibleSection("摄像头")
        camera_form = QFormLayout()
        camera_form.addRow("摄像头编号：", self.camera_combo)
        camera_form.addRow("ROI 尺寸：", self.roi_size_spin)
        camera_section.body_layout().addLayout(camera_form)
        camera_section.body_layout().addWidget(self.open_camera_button)

        self.pause_button = QPushButton("暂停")
        self.stop_stream_button = QPushButton("停止")
        self.save_video_button = QPushButton("保存结果")
        decorate(self.pause_button, "secondaryButton", QStyle.SP_MediaPause)
        decorate(self.stop_stream_button, "dangerButton", QStyle.SP_MediaStop)
        decorate(self.save_video_button, "secondaryButton", QStyle.SP_DialogSaveButton)
        playback_section = CollapsibleSection("播放控制")
        playback_row = QHBoxLayout()
        playback_row.addWidget(self.pause_button)
        playback_row.addWidget(self.stop_stream_button)
        playback_section.body_layout().addLayout(playback_row)
        playback_section.body_layout().addWidget(self.save_video_button)

        content = QWidget()
        content.setObjectName("leftPanelContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 16)
        content_layout.setSpacing(12)
        for section in (model_section, image_section, video_section, camera_section, playback_section):
            content_layout.addWidget(section)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("leftPanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setMinimumWidth(270)
        return scroll

    def _build_center_panel(self):
        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_image_page())
        self.mode_stack.addWidget(self._build_video_page())
        self.mode_stack.addWidget(self._build_camera_page())

        self.image_mode_button = QPushButton("图片模式")
        self.video_mode_button = QPushButton("视频模式")
        self.camera_mode_button = QPushButton("摄像头模式")
        self.mode_buttons = [self.image_mode_button, self.video_mode_button, self.camera_mode_button]

        switch_row = QHBoxLayout()
        for index, button in enumerate(self.mode_buttons):
            button.setObjectName("modeButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(38)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, i=index: self._set_mode(i))
            switch_row.addWidget(button)
        switch_row.addStretch(1)
        self.image_mode_button.setChecked(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addLayout(switch_row)
        layout.addWidget(self.mode_stack, 1)
        return container

    def _build_image_page(self):
        ensure_placeholder()
        placeholder_pixmap = QPixmap(str(PLACEHOLDER_PATH))
        if placeholder_pixmap.isNull():
            placeholder_pixmap = None

        self.original_canvas = ImageCanvas(
            "请选择图片", placeholder_pixmap=placeholder_pixmap
        )
        self.preprocessed_canvas = ImageCanvas("预处理图将在这里显示")
        self.overlay_canvas = ImageCanvas("识别结果叠加图将在这里显示")

        top_splitter = QSplitter(Qt.Horizontal)
        for title, canvas in (("原图", self.original_canvas), ("预处理后灰度图", self.preprocessed_canvas)):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(canvas)
            top_splitter.addWidget(box)
        top_splitter.setSizes([450, 450])

        bottom_box = QGroupBox("预测结果叠加")
        bottom_layout = QVBoxLayout(bottom_box)
        bottom_layout.addWidget(self.overlay_canvas)

        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(top_splitter)
        vertical_splitter.addWidget(bottom_box)
        vertical_splitter.setSizes([300, 300])

        page = QWidget()
        QVBoxLayout(page).addWidget(vertical_splitter)
        return page

    def _build_video_page(self):
        self.video_canvas = ImageCanvas("请加载模型后选择 MP4/AVI 视频", 500)
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_progress.setFormat("视频进度：%p%")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.video_canvas, 1)
        layout.addWidget(self.video_progress)
        return page

    def _build_camera_page(self):
        self.camera_canvas = ImageCanvas("请选择摄像头编号并点击“打开摄像头”", 520)
        hint = QLabel("中心红框为固定 ROI，仅对 ROI 内的交通标志进行分类。")
        hint.setAlignment(Qt.AlignCenter)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.camera_canvas, 1)
        layout.addWidget(hint)
        return page

    def _build_right_panel(self):
        result_box = QGroupBox("识别结果")
        result_box.setObjectName("resultCard")
        form = QFormLayout(result_box)
        form.setContentsMargins(16, 18, 16, 16)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        self.class_id_value = QLabel("--")
        self.class_id_value.setObjectName("resultValue")
        self.class_name_value = FadingLabel("--")
        self.class_name_value.setObjectName("resultName")
        self.class_name_value.setWordWrap(True)
        self.confidence_value = ConfidenceBar()
        form.addRow("当前类别 ID：", self.class_id_value)
        form.addRow("类别名：", self.class_name_value)
        form.addRow("置信度：", self.confidence_value)

        topk_box = QGroupBox("Top-3 预测")
        topk_box.setObjectName("topkCard")
        topk_layout = QVBoxLayout(topk_box)
        topk_layout.setContentsMargins(16, 18, 16, 16)
        topk_layout.setSpacing(10)
        self.topk_name_labels = []
        self.topk_bars = []
        for _ in range(3):
            row = QHBoxLayout()
            name_label = QLabel("--")
            name_label.setMinimumWidth(90)
            bar = QProgressBar()
            bar.setObjectName(f"top{len(self.topk_bars) + 1}Bar")
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            bar.setMinimumHeight(20)
            row.addWidget(name_label)
            row.addWidget(bar, 1)
            topk_layout.addLayout(row)
            self.topk_name_labels.append(name_label)
            self.topk_bars.append(bar)

        history_box = QGroupBox("历史记录")
        history_box.setObjectName("historyCard")
        history_layout = QVBoxLayout(history_box)
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.clear_history_button = QPushButton("清空历史")
        self.clear_history_button.setObjectName("dangerButton")
        self.clear_history_button.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        self.clear_history_button.setCursor(Qt.PointingHandCursor)
        self.clear_history_button.setMinimumHeight(36)
        history_layout.addWidget(self.history_list, 1)
        history_layout.addWidget(self.clear_history_button)

        self.dashboard = DashboardWidget()
        self.dashboard.set_theme(self.current_theme)

        panel = QWidget()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(result_box)
        layout.addWidget(topk_box)
        layout.addWidget(history_box, 3)
        layout.addWidget(self.dashboard, 2)
        panel.setMinimumWidth(330)
        return panel

    def _create_status_bar(self):
        status = QStatusBar(self)
        self.setStatusBar(status)
        self.operation_status_label = QLabel("就绪")
        self.model_status_label = QLabel("当前模型：未加载")
        self.feature_status_label = QLabel("特征模式：--")
        self.fps_status_label = QLabel("FPS：--")
        self.time_status_label = QLabel("耗时：-- ms")
        self.scene_status_label = QLabel("场景：--")
        self.operation_status_label.setObjectName("operationStatus")
        for label in (
            self.model_status_label,
            self.feature_status_label,
            self.fps_status_label,
            self.time_status_label,
            self.scene_status_label,
        ):
            label.setObjectName("statusMetric")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumWidth(90)
        status.addWidget(self.operation_status_label, 1)
        status.addPermanentWidget(self.model_status_label)
        status.addPermanentWidget(self.feature_status_label)
        status.addPermanentWidget(self.fps_status_label)
        status.addPermanentWidget(self.time_status_label)
        status.addPermanentWidget(self.scene_status_label)
        self.theme_toggle_button = QToolButton()
        self.theme_toggle_button.setObjectName("themeToggle")
        self.theme_toggle_button.setFixedSize(28, 28)
        self.theme_toggle_button.clicked.connect(self._toggle_theme)
        status.addPermanentWidget(self.theme_toggle_button)
        self._update_theme_controls()

    def _connect_controls(self):
        pairs = [
            (self.browse_model_button, self.browse_model),
            (self.load_model_button, self.load_model),
            (self.choose_image_button, self.choose_image),
            (self.predict_button, self.start_prediction),
            (self.scene_detect_button, self.start_scene_detection),
            (self.clear_button, self.clear_all),
            (self.select_video_button, self.choose_video),
            (self.open_camera_button, self.open_camera),
            (self.pause_button, self.toggle_pause),
            (self.stop_stream_button, self.stop_stream),
            (self.save_video_button, self.toggle_save_result),
            (self.clear_history_button, self._clear_history),
        ]
        for button, slot in pairs:
            button.clicked.connect(slot)
        self.model_path_edit.textChanged.connect(self._update_action_states)
        self.adaptive_enhance_checkbox.toggled.connect(self._on_adaptive_toggled)

    def _on_adaptive_toggled(self, enabled: bool) -> None:
        self._apply_adaptive_to_predictor()
        if self.active_recognizer is not None:
            self.active_recognizer.adaptive = bool(enabled)
        if self.current_image is not None:
            self._show_preprocessed_image()
            self._update_scene_status(self.scene_analyzer.analyze(self.current_image))
        self.operation_status_label.setText(
            "自适应增强已开启" if enabled else "自适应增强已关闭"
        )

    def _apply_adaptive_to_predictor(self) -> None:
        if self.predictor is None:
            return
        enabled = self.adaptive_enhance_checkbox.isChecked()
        preprocessor = getattr(self.predictor, "preprocessor", None)
        if preprocessor is not None:
            if hasattr(preprocessor, "set_adaptive"):
                preprocessor.set_adaptive(enabled)
            else:
                preprocessor.adaptive = enabled
            if enabled and self.current_image is not None and hasattr(preprocessor, "set_runtime_params"):
                analysis = self.scene_analyzer.analyze(self.current_image)
                preprocessor.set_runtime_params(**self.scene_analyzer.recommend_params(analysis))
        if hasattr(self.predictor, "clear_cache"):
            self.predictor.clear_cache()

    def _update_scene_status(self, analysis: dict) -> None:
        brightness = float(analysis.get("brightness", 0.0))
        blur_score = float(analysis.get("blur_score", 0.0))
        degradations = set(analysis.get("degradations", ()))
        light_icon = "\u263e" if "low_light" in degradations else "\u2600"
        blur_icon = "\u2248" if "blur" in degradations else "\u25c6"
        self.scene_status_label.setText(
            f"场景：{light_icon} {brightness:.0f} / {blur_icon} {blur_score:.0f}"
        )
        names = {
            "low_light": "低光照",
            "fog": "低对比/雾",
            "blur": "模糊",
            "noise": "噪声",
        }
        detail = "、".join(names.get(item, item) for item in analysis.get("degradations", ()))
        self.scene_status_label.setToolTip(
            f"亮度 {brightness:.1f}；对比度 {float(analysis.get('contrast', 0.0)):.1f}；"
            f"Laplacian 方差 {blur_score:.1f}；退化：{detail or '无明显退化'}"
        )

    def _set_mode(self, index: int) -> None:
        self.mode_stack.setCurrentIndex(index)
        for i, button in enumerate(self.mode_buttons):
            button.setChecked(i == index)

    def _toggle_theme(self, checked=False) -> None:
        """Switch the application QSS immediately and refresh theme affordances."""
        del checked
        app = QApplication.instance()
        if app is None:
            return
        self.current_theme = toggle_theme(app, self.current_theme)
        self._update_theme_controls()
        self.confidence_value.update()
        self.dashboard.set_theme(self.current_theme)

    def _update_theme_controls(self) -> None:
        is_dark = self.current_theme == "dark"
        self.theme_toggle_button.setText("☀" if is_dark else "☾")
        self.theme_toggle_button.setToolTip(
            "切换为亮色主题 (Ctrl+T)" if is_dark else "切换为暗色主题 (Ctrl+T)"
        )
        self.toggle_theme_action.setText("切换亮色主题" if is_dark else "切换暗色主题")

    def _toggle_left_panel(self, checked=True):
        self.left_panel.setVisible(checked)

    def _toggle_right_panel(self, checked=True):
        self.right_panel.setVisible(checked)

    def _set_ready_state(self):
        model = self._find_default_model()
        if model:
            self.model_path_edit.setText(str(model))
        self._update_action_states()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _try_load_model(self, path: Path) -> None:
        """Public helper: set the model path and trigger background loading.

        Called by ``main.py`` on startup when the default bundle exists.
        """
        raw = str(path.resolve())
        if not path.is_file():
            logger.warning("_try_load_model: file not found – %s", raw)
            return
        self.model_path_edit.setText(raw)
        self.load_model()

    @staticmethod
    def _find_default_model():
        folder = Path(__file__).resolve().parents[1] / "models" / "artifacts"
        files = sorted(folder.glob("*.joblib"))
        return files[0] if files else None

    def browse_model(self):
        initial = self.model_path_edit.text().strip() or str(
            Path(__file__).resolve().parents[1] / "models" / "artifacts"
        )
        path, _ = QFileDialog.getOpenFileName(self, "选择模型 Bundle", initial, self.MODEL_FILTER)
        if path:
            self.model_path_edit.setText(path)

    def load_model(self):
        if self._thread_is_running(self.load_worker):
            return
        if self.active_recognizer is not None:
            self._show_error("加载模型失败", "请先停止当前视频或摄像头。")
            return
        raw = self.model_path_edit.text().strip()
        if not raw:
            self._show_error("加载模型失败", "请先选择模型文件。")
            return
        path = Path(raw).expanduser()
        if not path.is_file():
            self._show_error("加载模型失败", f"模型文件不存在：\n{path}")
            return
        self.predictor = None
        self._loading_bundle_path = path
        self.operation_status_label.setText("正在后台加载模型…")
        self.loading_overlay.show_loading()
        self.load_worker = LoadModelWorker(path, self)
        self.load_worker.loaded.connect(self._on_model_loaded)
        self.load_worker.error.connect(lambda m: self._show_error("加载模型失败", m))
        self.load_worker.finished.connect(self._on_load_worker_finished)
        self.load_worker.start()
        self._update_action_states()

    def _on_model_loaded(self, predictor):
        self.predictor = predictor
        self._apply_adaptive_to_predictor()
        summary = predictor.summary
        model = str(summary.get("model") or predictor.classifier.__class__.__name__)
        mode = str(predictor.feature_config.get("mode", summary.get("feature_mode", "--")))
        self.model_status_label.setText(f"当前模型：{model}")
        self.feature_status_label.setText(f"特征模式：{mode}")
        self.operation_status_label.setText("模型加载成功")
        self.dashboard.set_confusion_matrix(self._loading_bundle_path)
        if self.current_image is not None:
            self._show_preprocessed_image()
        self._update_action_states()

    def _on_load_worker_finished(self):
        worker = self.load_worker
        self.load_worker = None
        if worker:
            worker.deleteLater()
        self.loading_overlay.hide_loading()
        if self.predictor is None:
            self.operation_status_label.setText("模型未加载")
        self._update_action_states()

    # ------------------------------------------------------------------
    # Image mode
    # ------------------------------------------------------------------
    def choose_image(self):
        initial = str(self.current_image_path.parent if self.current_image_path else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "选择交通标志图片", initial, self.IMAGE_FILTER)
        if not path:
            return
        try:
            image = self._read_image(Path(path))
        except Exception as exc:
            self._show_error("打开图片失败", f"{type(exc).__name__}: {exc}")
            return
        self.current_image = image
        self.current_image_path = Path(path)
        self.original_canvas.set_image(numpy_to_pixmap(image))
        self.overlay_canvas.clear_image()
        self._reset_result_labels()
        self._clear_topk()
        self._show_preprocessed_image()
        self._update_scene_status(self.scene_analyzer.analyze(image))
        self._set_mode(0)
        self.operation_status_label.setText(f"已选择图片：{self.current_image_path.name}")
        self._update_action_states()

    @staticmethod
    def _read_image(path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"图片文件不存在：{path}")
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("无法解码该图片，请确认文件格式和内容有效。")
        return image

    def _show_preprocessed_image(self):
        if self.current_image is None:
            self.preprocessed_canvas.clear_image()
            return
        try:
            if self.predictor is not None and self.predictor.preprocessor is not None:
                processed = self.predictor.preprocessor(self.current_image)
            else:
                size = int(self.predictor.feature_config.get("img_size", 64)) if self.predictor else 64
                processed = cv2.cvtColor(cv2.resize(self.current_image, (size, size)), cv2.COLOR_BGR2GRAY)
            if processed.ndim == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            self.preprocessed_canvas.set_image(numpy_to_pixmap(processed))
        except Exception as exc:
            self.preprocessed_canvas.clear_image()
            self._show_error("预处理失败", f"{type(exc).__name__}: {exc}")

    def start_prediction(self):
        if self._thread_is_running(self.predict_worker):
            return
        if self.active_recognizer is not None:
            self._show_error("无法识别", "请先停止当前视频或摄像头。")
            return
        if self.predictor is None:
            self._show_error("无法识别", "请先加载模型。")
            return
        if self.current_image is None:
            self._show_error("无法识别", "请先选择图片。")
            return
        self.operation_status_label.setText("正在后台识别图片…")
        self.predict_worker = PredictWorker(self.predictor, self.current_image, self)
        self.predict_worker.predicted.connect(self._on_image_predicted)
        self.predict_worker.top_k.connect(self._on_topk_ready)
        self.predict_worker.error.connect(lambda m: self._show_error("识别失败", m))
        self.predict_worker.finished.connect(self._on_predict_worker_finished)
        self.predict_worker.start()
        self._update_action_states()

    def _on_image_predicted(self, result):
        self._update_result_labels(result)
        if self.current_image is not None:
            overlay = overlay_prediction(self.current_image, result)
            self.overlay_canvas.set_image(numpy_to_pixmap(overlay))
        self.operation_status_label.setText(f"识别完成：{result['class_name']}")
        self._add_history_entry(self.current_image_path, result)

    def _on_topk_ready(self, items):
        for index, (name_label, bar) in enumerate(zip(self.topk_name_labels, self.topk_bars)):
            if index < len(items):
                item = items[index]
                confidence = item.get("confidence") or 0.0
                name_label.setText(str(item.get("class_name", "--")))
                bar.setValue(int(round(confidence * 100)))
            else:
                name_label.setText("--")
                bar.setValue(0)
        prediction = getattr(self, "_latest_prediction", None)
        self.dashboard.update_confidence(
            {
                str(item.get("class_name", "--")): float(item.get("confidence", 0.0))
                for item in items
            },
            prediction.get("class_name") if prediction else None,
        )

    def _add_history_entry(self, path, result) -> None:
        name = Path(path).name if path else "当前图片"
        confidence = result.get("confidence")
        conf_text = "--" if confidence is None else f"{float(confidence):.1%}"
        self.history_list.insertItem(0, f"{name} → {result['class_name']} ({conf_text})")

    def _clear_topk(self) -> None:
        for name_label, bar in zip(self.topk_name_labels, self.topk_bars):
            name_label.setText("--")
            bar.setValue(0)

    def _clear_history(self) -> None:
        self.history_list.clear()
        self.dashboard.reset_history()

    def _on_predict_worker_finished(self):
        worker = self.predict_worker
        self.predict_worker = None
        if worker:
            worker.deleteLater()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Scene detection (image or last stream frame)
    # ------------------------------------------------------------------
    def start_scene_detection(self):
        """Run SignDetector on the currently displayed image."""
        if self._thread_is_running(self.detect_worker):
            return
        if self.active_recognizer is not None and self.stream_mode is None:
            self._show_error("无法检测", "请先停止当前视频或摄像头。")
            return
        if self.predictor is None:
            self._show_error("无法检测", "请先加载模型。")
            return
        target = self.current_image
        if target is None:
            if self.active_recognizer is not None and self.last_stream_frame is not None:
                target = self.last_stream_frame
            else:
                self._show_error("无法检测", "请先选择图片或打开视频。")
                return
        self.operation_status_label.setText("正在后台进行场景检测…")
        detector = SignDetector(self.predictor)
        self.detect_worker = DetectWorker(detector, target, self)
        self.detect_worker.detected.connect(self._on_scene_detected)
        self.detect_worker.annotated.connect(self._on_scene_annotated)
        self.detect_worker.error.connect(lambda m: self._show_error("场景检测失败", m))
        self.detect_worker.finished.connect(self._on_detect_worker_finished)
        self.detect_worker.start()
        self._update_action_states()

    def _on_scene_detected(self, results):
        count = len(results)
        self.operation_status_label.setText(f"场景检测完成：发现 {count} 个候选标志")

    def _on_scene_annotated(self, annotated_img):
        if self.stream_mode is None:
            self.overlay_canvas.set_image(numpy_to_pixmap(annotated_img))
        elif self.stream_mode == "video" and self.last_stream_frame is not None:
            self.video_canvas.set_image(numpy_to_pixmap(annotated_img))
        elif self.stream_mode == "camera" and self.last_stream_frame is not None:
            self.camera_canvas.set_image(numpy_to_pixmap(annotated_img))

    def _on_detect_worker_finished(self):
        worker = self.detect_worker
        self.detect_worker = None
        if worker:
            worker.deleteLater()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def start_batch_predict(self):
        if self._thread_is_running(self.batch_worker):
            return
        if not self._require_predictor("批量识别"):
            return
        if self.active_recognizer is not None:
            self._show_error("无法批量识别", "请先停止当前视频或摄像头。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(Path.cwd()))
        if not folder:
            return
        folder_path = Path(folder)
        extensions = BatchPredictWorker._EXTENSIONS
        files = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
        if not files:
            self._show_error("批量识别失败", "该文件夹内没有找到支持的图片文件。")
            return

        dialog = QProgressDialog("正在批量识别…", "取消", 0, len(files), self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        self.batch_worker = BatchPredictWorker(self.predictor, folder_path, self)
        self.batch_worker.progress.connect(lambda done, total: dialog.setValue(done))
        dialog.canceled.connect(self.batch_worker.cancel)
        self.batch_worker.finished_batch.connect(self._on_batch_finished)
        self.batch_worker.error.connect(lambda m: self._show_error("批量识别失败", m))
        self.batch_worker.finished.connect(lambda: self._on_batch_worker_finished(dialog))
        self.operation_status_label.setText(f"正在批量识别：{folder_path.name}")
        self.batch_worker.start()
        self._update_action_states()

    def _on_batch_finished(self, results):
        for entry in results:
            name = Path(entry["path"]).name
            if entry.get("error"):
                text = f"{name} → 识别失败：{entry['error']}"
            else:
                confidence = entry.get("confidence")
                conf_text = "--" if confidence is None else f"{float(confidence):.1%}"
                text = f"{name} → {entry['class_name']} ({conf_text})"
            self.history_list.insertItem(0, text)
        ok_count = sum(1 for entry in results if not entry.get("error"))
        self.operation_status_label.setText(f"批量识别完成：成功 {ok_count}/{len(results)}")

    def _on_batch_worker_finished(self, dialog):
        worker = self.batch_worker
        self.batch_worker = None
        if worker:
            worker.deleteLater()
        dialog.close()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Video mode
    # ------------------------------------------------------------------
    def choose_video(self):
        if not self._require_predictor("打开视频"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", str(Path.cwd()), self.VIDEO_FILTER)
        if path:
            self.open_video_path(path)

    def open_video_path(self, path):
        if not self._require_predictor("打开视频"):
            return False
        path = Path(path)
        if not path.is_file():
            self._show_error("打开视频失败", f"视频文件不存在：\n{path}")
            return False
        self.stop_stream(silent=True)
        recognizer = VideoRecognizer(
            self.predictor, path, adaptive=self.adaptive_enhance_checkbox.isChecked()
        )
        if not recognizer.open():
            self._show_error("打开视频失败", "无法打开该视频，请检查格式或编码。")
            return False
        self.active_recognizer = recognizer
        self.stream_mode = "video"
        self.stream_paused = False
        self.last_stream_frame = None
        self.video_canvas.clear_image()
        self._clear_topk()
        self._set_mode(1)
        if recognizer.frame_count > 0:
            self.video_progress.setRange(0, recognizer.frame_count)
            self.video_progress.setValue(0)
            self.video_progress.setFormat("视频进度：%v / %m 帧（%p%）")
        else:
            self.video_progress.setRange(0, 0)
            self.video_progress.setFormat("正在读取视频…")
        self.operation_status_label.setText(f"正在播放：{path.name}")
        self.fps_status_label.setText("FPS：--")
        self.time_status_label.setText("耗时：-- ms")
        self._set_pause_text(False)
        self.stream_timer.start()
        self._update_action_states()
        return True

    def open_camera(self):
        if not self._require_predictor("打开摄像头"):
            return False
        index = int(self.camera_combo.currentText())
        roi_size = self.roi_size_spin.value()
        self.stop_stream(silent=True)
        recognizer = CameraRecognizer(
            self.predictor, index, roi_size,
            adaptive=self.adaptive_enhance_checkbox.isChecked(),
        )
        if not recognizer.open():
            recognizer.release()
            self._show_error("打开摄像头失败", "未检测到摄像头。")
            return False
        self.active_recognizer = recognizer
        self.stream_mode = "camera"
        self.stream_paused = False
        self.last_stream_frame = None
        self.camera_canvas.clear_image()
        self._clear_topk()
        self._set_mode(2)
        self.operation_status_label.setText(f"摄像头 {index} 已打开")
        self.fps_status_label.setText("FPS：--")
        self.time_status_label.setText("耗时：-- ms")
        self._set_pause_text(False)
        self.stream_timer.start()
        self._update_action_states()
        return True

    def _on_tick(self):
        recognizer = self.active_recognizer
        if recognizer is None:
            self.stream_timer.stop()
            return
        try:
            ok, frame, result = recognizer.read()
        except Exception as exc:
            self._show_error("视频识别失败", f"{type(exc).__name__}: {exc}")
            self.stop_stream(silent=True)
            return
        if not ok or frame is None:
            if self.stream_mode == "video":
                self._finish_stream("视频播放结束")
            else:
                self._show_error("摄像头读取失败", "无法继续读取摄像头画面。")
                self.stop_stream(silent=True)
            return
        self.last_stream_frame = frame.copy()
        if self.stream_mode == "video":
            self.video_canvas.set_image(numpy_to_pixmap(frame))
            if recognizer.frame_count > 0:
                self.video_progress.setValue(min(recognizer.frame_index, recognizer.frame_count))
        else:
            self.camera_canvas.set_image(numpy_to_pixmap(frame))
        self._update_result_labels(result)
        ms = float(result.get("predict_seconds", 0)) * 1000
        fps = float(result.get("fps", 0))
        self.fps_status_label.setText(f"FPS：{fps:.1f}")
        self.time_status_label.setText(f"耗时：{ms:.1f} ms")
        mode_text = "视频" if self.stream_mode == "video" else "摄像头"
        self.operation_status_label.setText(f"{mode_text}识别：{result.get('class_name', '--')}")
        self._write_output_frame(frame)

    def toggle_pause(self):
        if self.active_recognizer is None:
            return
        if self.stream_timer.isActive():
            self.stream_timer.stop()
            self.stream_paused = True
            self._set_pause_text(True)
            self.operation_status_label.setText("已暂停")
        else:
            self.stream_timer.start()
            self.stream_paused = False
            self._set_pause_text(False)
            self.operation_status_label.setText("继续识别")
        self._update_action_states()

    def _set_pause_text(self, paused):
        text = "继续" if paused else "暂停"
        self.pause_action.setText(text)
        if hasattr(self, "pause_button"):
            self.pause_button.setText(text)

    def stop_stream(self, checked=False, silent=False):
        del checked
        if self.active_recognizer is None and self.video_writer is None and self.pending_save_path is None:
            return
        self.stream_timer.stop()
        if self.active_recognizer is not None:
            self.active_recognizer.release()
        self.active_recognizer = None
        self.stream_mode = None
        self.stream_paused = False
        self._finish_writer()
        self._set_pause_text(False)
        if not silent:
            self.operation_status_label.setText("视频/摄像头已停止")
        self._update_action_states()

    def _finish_stream(self, message):
        self.stream_timer.stop()
        if self.active_recognizer is not None:
            self.active_recognizer.release()
        saved = self.output_save_path or self.pending_save_path
        self.active_recognizer = None
        self.stream_mode = None
        self.stream_paused = False
        self._finish_writer()
        self._set_pause_text(False)
        self.operation_status_label.setText(message + (f"；结果已保存至 {saved.name}" if saved else ""))
        self._update_action_states()

    def toggle_save_result(self):
        if self.video_writer is not None or self.pending_save_path is not None:
            saved = self.output_save_path or self.pending_save_path
            self._finish_writer()
            self.operation_status_label.setText(f"已停止保存：{saved}")
            self._update_action_states()
            return
        if self.active_recognizer is None:
            self._show_error("无法保存", "请先打开视频或摄像头。")
            return
        name = "recognition_result.mp4"
        if isinstance(self.active_recognizer.src, (str, Path)):
            source = Path(self.active_recognizer.src)
            name = f"{source.stem}_result.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存识别结果", str(Path.cwd() / name), "MP4 视频 (*.mp4);;AVI 视频 (*.avi)"
        )
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() not in {".mp4", ".avi"}:
            output = output.with_suffix(".mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.pending_save_path = output
        self.output_save_path = None
        self.operation_status_label.setText(f"准备保存识别结果：{output.name}")
        self._update_action_states()

    def _write_output_frame(self, frame):
        if self.pending_save_path is None and self.video_writer is None:
            return
        if self.video_writer is None:
            height, width = frame.shape[:2]
            suffix = self.pending_save_path.suffix.lower()
            codec = "XVID" if suffix == ".avi" else "mp4v"
            fps = 30.0
            if (
                self.active_recognizer is not None
                and np.isfinite(self.active_recognizer.source_fps)
                and self.active_recognizer.source_fps > 1
            ):
                fps = self.active_recognizer.source_fps
            writer = cv2.VideoWriter(
                str(self.pending_save_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
            if not writer.isOpened():
                writer.release()
                failed = self.pending_save_path
                self.pending_save_path = None
                self._show_error("保存失败", f"无法创建输出视频：\n{failed}")
                self._update_action_states()
                return
            self.video_writer = writer
            self.output_save_path = self.pending_save_path
            self.pending_save_path = None
        self.video_writer.write(np.ascontiguousarray(frame))

    def _finish_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        self.pending_save_path = None
        self.output_save_path = None

    def _require_predictor(self, action):
        if self.predictor is None:
            self._show_error(f"无法{action}", "请先加载模型。")
            return False
        return True

    def _update_result_labels(self, result):
        self._latest_prediction = result
        self.class_id_value.setText(str(int(result["class_id"])))
        self.class_name_value.setText(str(result["class_name"]))
        confidence = result.get("confidence")
        if confidence is None:
            self.confidence_value.setUnavailable()
            self.confidence_value.setToolTip("本模型不支持概率输出")
        else:
            self.confidence_value.setToolTip(f"{float(confidence):.2%}")
            self.confidence_value.setValue(float(confidence) * 100)
        self.dashboard.update_confidence(
            {} if confidence is None else {str(result["class_name"]): float(confidence)},
            str(result["class_name"]),
        )
        self.dashboard.append_prediction(int(result["class_id"]), confidence)

    def clear_all(self):
        if self._thread_is_running(self.predict_worker):
            self._show_error("暂时无法清空", "图片识别任务正在运行，请稍候。")
            return
        self.current_image = None
        self.current_image_path = None
        self.original_canvas.clear_image()
        self.preprocessed_canvas.clear_image()
        self.overlay_canvas.clear_image()
        self._reset_result_labels()
        self._clear_topk()
        self.dashboard.reset_history()
        self.operation_status_label.setText("图片区域已清空")
        self._update_action_states()

    def _reset_result_labels(self):
        self.class_id_value.setText("--")
        self.class_name_value.setText("--")
        self.confidence_value.reset()
        self.confidence_value.setToolTip("")

    def _update_action_states(self):
        loading = self._thread_is_running(self.load_worker)
        predicting = self._thread_is_running(self.predict_worker)
        detecting = self._thread_is_running(self.detect_worker)
        batching = self._thread_is_running(self.batch_worker)
        streaming = self.active_recognizer is not None
        busy = loading or predicting or detecting or batching
        has_path = bool(self.model_path_edit.text().strip())
        has_model = self.predictor is not None
        saving = self.video_writer is not None or self.pending_save_path is not None

        for widget in [self.model_path_edit, self.browse_model_button, self.browse_model_action]:
            widget.setEnabled(not busy and not streaming)
        self.load_model_button.setEnabled(has_path and not busy and not streaming)
        self.load_model_action.setEnabled(has_path and not busy and not streaming)
        for widget in [self.choose_image_button, self.open_image_action, self.clear_button, self.clear_action]:
            widget.setEnabled(not busy and not streaming)

        can_image = has_model and self.current_image is not None and not busy and not streaming
        self.predict_button.setEnabled(can_image)
        self.predict_action.setEnabled(can_image)

        can_detect = (
            has_model
            and (self.current_image is not None or (streaming and self.last_stream_frame is not None))
            and not busy
        )
        self.scene_detect_button.setEnabled(can_detect)
        self.scene_detect_action.setEnabled(can_detect)

        can_stream = has_model and not busy
        self.select_video_button.setEnabled(can_stream)
        self.select_video_action.setEnabled(can_stream)
        self.open_camera_button.setEnabled(can_stream)
        self.open_camera_action.setEnabled(can_stream)
        self.camera_combo.setEnabled(not busy and not streaming)
        self.roi_size_spin.setEnabled(not busy and not streaming)
        self.adaptive_enhance_checkbox.setEnabled(not busy)

        can_batch = has_model and not busy and not streaming
        self.batch_predict_action.setEnabled(can_batch)

        for widget in [
            self.pause_button,
            self.pause_action,
            self.stop_stream_button,
            self.stop_stream_action,
            self.save_video_button,
            self.save_video_action,
        ]:
            widget.setEnabled(streaming)

        text = "停止保存" if saving else "保存结果"
        self.save_video_button.setText(text)
        self.save_video_action.setText(text)

    @staticmethod
    def _thread_is_running(worker):
        return worker is not None and worker.isRunning()

    def _show_error(self, title, message):
        QMessageBox.critical(self, title, f"操作失败：\n{message}")
        self.operation_status_label.setText(title)

    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "交通标志分类识别系统\n\n"
            "支持图片、视频和中心 ROI 摄像头识别。\n"
            "视频帧由 33ms QTimer 驱动，可暂停、继续和保存。",
        )

    def closeEvent(self, event):
        if (
            self._thread_is_running(self.load_worker)
            or self._thread_is_running(self.predict_worker)
            or self._thread_is_running(self.detect_worker)
            or self._thread_is_running(self.batch_worker)
        ):
            QMessageBox.warning(self, "任务正在运行", "后台任务尚未结束，请稍候再关闭窗口。")
            event.ignore()
            return
        self.stop_stream(silent=True)
        event.accept()

