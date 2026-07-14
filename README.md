
﻿# 交通标志分类识别系统

基于 **OpenCV + HOG/HSV + SVM / KNN / RandomForest** 的交通标志分类识别的项目。
覆盖数据加载 → 预处理 → 特征抽取 → 多分类器训练 → 评估 → 推理加速 → 桌面 GUI 全流程。

---

## 项目目标

1. 使用 OpenCV 对交通标志图像进行预处理（灰度化、CLAHE、尺寸归一化）。
2. 通过 HOG（方向梯度直方图）+ HSV 颜色直方图 提取图像特征。
3. 使用 **SVM / KNN / RandomForest / 软投票集成** 多分类器对比与组合。
4. 引入 **类别权重平衡、数据增强、贝叶斯超参搜索** 系统性提升精度。
5. 提供 **特征缓存、批量预测、帧间跳过** 等推理加速策略，单图 < 50 ms、视频 ≥ 15 FPS。
6. 提供可视化界面（PyQt）方便实时检测与结果展示。

---

## 快速开始

| 任务 | 命令 |
| --- | --- |
| 安装依赖 | `pip install -r requirements.txt` |
| 训练 SVM (HOG+HSV, balanced) | `python -m traffic_sign_system.scripts.train --model svm --mode hog+hsv --class-weight balanced` |
| 训练 SVM+KNN+RF 集成 | `python -m traffic_sign_system.scripts.train --mode hog+hsv --ensemble` |
| 评估 bundle | `python -m traffic_sign_system.scripts.evaluate --bundle models/artifacts/svm_hog+hsv.joblib` |
| 单图预测 | `python -m traffic_sign_system.scripts.predict_one --bundle <bundle> --image <png>` |
| 启动桌面 GUI | `python -m traffic_sign_system` |
| 推理性能基准 | `python -m traffic_sign_system.scripts.benchmark --bundle <bundle>` |
| 贝叶斯超参搜索 | `python -m traffic_sign_system.scripts.hyperopt_search --n-calls 30` |

---

## 目录结构

```
traffic_sign_system/
├── main.py                 # 桌面 GUI 入口
├── __main__.py             # python -m traffic_sign_system 启动器
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
│
├── config/
│   ├── settings.py         # 路径、HOG 参数、SVM 超参、数据划分比例
│   └── labels.py           # 类别 ID → 名称映射（支持从 labels.csv 加载）
│
├── data_processing/
│   ├── data_loader.py      # 训练/测试数据加载
│   ├── preprocessing.py    # BGR → 灰度 → 去噪 → CLAHE → 归一化
│   ├── augmentation.py     # 数据增强池（affine / perspective / motion_blur / cutout / strong）
│   └── dataset_statistics.py
│
├── features/
│   ├── hog_extractor.py    # OpenCV HOGDescriptor 封装
│   ├── color_extractor.py  # HSV 颜色直方图
│   └── feature_fusion.py   # FeatureBuilder：hog / hsv / hog+hsv 三种模式
│
├── models/
│   ├── train_svm.py        # SVM 训练（支持 class_weight=balanced）
│   ├── train_knn.py        # KNN 训练
│   ├── train_random_forest.py
│   ├── train_ensemble.py   # 软投票集成（SVM+KNN+RF）
│   ├── model_manager.py    # bundle 序列化 / 校验
│   └── artifacts/          # .joblib 模型包 + 历史 + 基准输出
│
├── evaluation/
│   ├── evaluator.py        # metrics.json / confusion_matrix / errors.csv
│   ├── confusion_matrix.py
│   ├── error_analysis.py
│   └── comparison.py       # SVM/KNN/RF 对比
│
├── recognition/            # 推理模块
│   ├── predictor.py        # 单图/批量预测 + LRU 缓存
│   ├── sign_detector.py    # HSV 颜色掩膜 + 轮廓候选区检测
│   ├── video_recognizer.py # 视频流帧间跳过 + ROI
│   └── camera_recognizer.py# 摄像头中心 ROI + 置信度平滑
│
├── scripts/                # 命令行入口
│   ├── train.py            # 训练（含 --class-weight / --ensemble / --grid-search）
│   ├── evaluate.py
│   ├── compare.py          # SVM/KNN/RF 对比
│   ├── error_stats.py      # 错误分析（top_confusions / per-class recall）
│   ├── build_features.py
│   ├── predict_one.py
│   ├── check_dataset.py
│   └── benchmark.py        # 推理性能基准（单图/批量/视频 FPS）
│   └── hyperopt_search.py  # 贝叶斯超参搜索（skopt）
│
└── ui/                     # PyQt 桌面界面
    ├── main_window.py
    ├── dashboard.py
    ├── widgets.py
    ├── workers.py
    ├── image_utils.py
    └── styles.py
```

---

## 环境安装

```bash
# 推荐使用 conda 或 venv 创建虚拟环境（Python 3.10 / 3.11 / 3.12）
conda create -n traffic_sign python=3.11 -y
conda activate traffic_sign

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 关键依赖：

```
opencv-python==4.10.0.84
PyQt5==5.15.10
Pillow>=10.0,<12
numpy<2
pandas>=2.2,<3
scikit-learn>=1.5
joblib>=1.4
scikit-optimize>=0.10        # 贝叶斯超参搜索
matplotlib>=3.10
```

---

## 训练

### 单模型

```bash
# SVM + HOG+HSV, balanced 类别权重（推荐主流程）
python -m traffic_sign_system.scripts.train \
    --model svm --mode hog+hsv --class-weight balanced \
    --C 10.0 --gamma scale --kernel rbf

# KNN
python -m traffic_sign_system.scripts.train --model knn --mode hog+hsv --n-neighbors 5

# Random Forest
python -m traffic_sign_system.scripts.train --model rf  --mode hog+hsv --n-estimators 200

# SVM + Grid Search（C / gamma 自动寻优）
python -m traffic_sign_system.scripts.train \
    --model svm --mode hog+hsv --grid-search \
    --C-grid 0.1 1.0 10.0 --gamma-grid scale 0.001 0.01 --cv-folds 3
```

### 软投票集成（SVM + KNN + RF，按 val accuracy 加权）

```bash
python -m traffic_sign_system.scripts.train \
    --mode hog+hsv --ensemble --class-weight balanced
```

训练完成后，bundle 自动保存到 `models/artifacts/{model|ensemble}_<mode>.joblib`，包含
classifier / scaler / label_map / feature_config / summary 五项元信息。

### 类别权重与样本平衡

| flag | 含义 |
| --- | --- |
| `--class-weight balanced` | SVM 按 `n_samples / (n_classes * np.bincount(y))` 自动平衡 43 类 |
| `--class-weight none` | 不做权重补偿 |
| `--max-samples N` | 等量下采样到 N 张以加速实验 |

GTSRB 类别 0/19/37 仅 ~210 张而 2/13 有 2000+ 张，`--class-weight balanced` 通常在少数类上
提升 recall 0.5~2%。

---

## 模型精度提升策略（第 ㉑ 步）

| 策略 | 文件 | 关键点 |
| --- | --- | --- |
| ① 类别权重平衡 | [models/train_svm.py](traffic_sign_system/models/train_svm.py) | `class_weight="balanced"` |
| ② 软投票集成 | [models/train_ensemble.py](traffic_sign_system/models/train_ensemble.py) | SVM+KNN+RF，`set_weights_from_scores` 按 val acc 加权 |
| ③ 数据增强调优 | [data_processing/augmentation.py](traffic_sign_system/data_processing/augmentation.py) | 新增 `random_perspective` / `motion_blur` / `cutout` / `apply_strong`；`AUGMENT_POOL` 7 项按权重采样 |
| ④ 贝叶斯超参搜索 | [scripts/hyperopt_search.py](traffic_sign_system/scripts/hyperopt_search.py) | `gp_minimize` 6 维空间，3-fold CV，写 `models/artifacts/hyperopt_history.json` |

```bash
# 策略 ④：默认 30 次迭代，10 个初始随机点
python -m traffic_sign_system.scripts.hyperopt_search \
    --mode hog+hsv --n-calls 30 --n-initial-points 10 \
    --output traffic_sign_system/models/artifacts/hyperopt_history.json
```

---

## 评估

### 单 bundle 评估

```bash
python -m traffic_sign_system.scripts.evaluate \
    --bundle "models/artifacts/svm_hog+hsv.joblib" \
    --data "dataset/test/" \
    --out "models/artifacts/eval/"
```

产出：

- `metrics.json` — 总体 accuracy / macro-F1 / weighted-F1 / 各类 PRF
- `confusion_matrix.png` — 归一化混淆矩阵
- `errors.csv` — 误判样本（含 `confidence` 列，分类器无 `predict_proba` 时为 `NaN`）

### 错误分析

```bash
python -m traffic_sign_system.scripts.error_stats \
    --bundle "models/artifacts/svm_hog+hsv.joblib" \
    --data "dataset/test/"
```

产出：

- `top_confusions.csv` / `.png` — 有向混淆对（A→B 不同于 B→A）
- `errors_per_class.csv` / `.png` — 各类 recall / error_count / support
- 同时复用 evaluator 写出 `metrics.json` / `confusion_matrix.png` / `errors.csv`

### SVM / KNN / RandomForest 对比

```bash
python -m traffic_sign_system.scripts.build_features --mode "hog+hsv"
python -m traffic_sign_system.scripts.compare \
    --features "models/artifacts/features_hog+hsv.npz"
```

产出 `comparison.csv`、`comparison.png`、`confusion_matrices.png`。

---

## 单图 / 批量预测

### Python API

```python
import cv2
from traffic_sign_system.recognition.predictor import Predictor

predictor = Predictor(
    "traffic_sign_system/models/artifacts/svm_hog+hsv.joblib",
    use_cache=True,    # LRU 缓存（视频相邻帧去重）
    cache_maxsize=512,
)

# 单图
img = cv2.imread("sample_64x64.png")
result = predictor.predict(img)
print(result)
# {'class_id': 12, 'class_name': 'Stop', 'confidence': 0.973214}

# 批量（一次 extract + scaler + predict，比循环快 3-5x）
imgs = [img] * 100
results = predictor.predict_batch(imgs)
print(predictor.cache_stats())
# {'hits': 99, 'misses': 1, 'hit_rate': 0.99, 'size': 1, 'maxsize': 512}
```

### 命令行

```bash
python -m traffic_sign_system.scripts.predict_one \
    --bundle "svm_hog+hsv.joblib" \
    --image "dataset/test/sample_64x64.png"
```

输出：

```text
class_id: 12
class_name: Stop
confidence: 0.973214
```

---

## 推理加速（第 ㉒ 步）

| 优化 | 实现位置 | 预期收益 |
| --- | --- | --- |
| LRU 图像缓存 | `Predictor(use_cache=True, cache_maxsize=512)` | 单图重复访问 ~25x 加速，视频 hit_rate > 80% |
| 批量预测 | `Predictor.predict_batch(imgs)` | 100 张批量 vs 100 次循环：3~5x 加速 |
| 帧间跳过 | `VideoRecognizer(skip_frames=2)` | FPS 提升 2~3x |
| 中心 ROI | `CameraRecognizer(roi_size=64)` | 减少特征抽取区域 |
| 置信度平滑 | `CameraRecognizer(smooth_window=3)` | 偶发噪声更稳健 |

```bash
# 性能基准：单图 / 批量 / 视频 FPS
python -m traffic_sign_system.scripts.benchmark \
    --bundle "models/artifacts/svm_hog+hsv.joblib" \
    --repeats-single 100 \
    --batch-sizes 1 10 50 100 \
    --skip-options 0 1 2 \
    --output "models/artifacts/benchmark.json"
```

控制台输出示例：

```text
========== BENCHMARK SUMMARY ==========
[1] Single predict
  cold (no cache)   : 40.21 ms/call
  warm (with cache) :  0.52 ms/call
  speedup           : 77.3x
  cache hit_rate    : 0.99
[2] Batch predict
  batch= 100  loop=4000 ms  batch=900 ms  speedup=4.4x
[3] Video FPS
    skip_0  fps_mean=22.1  reuse_rate=0.00  cache_hit=0.83
    skip_2  fps_mean=50.3  reuse_rate=0.65  cache_hit=0.79
```

---

## 桌面 GUI

```bash
# 自动加载 models/artifacts/svm_hog+hsv.joblib（如存在）
python -m traffic_sign_system
# 等价于 python -m traffic_sign_system.main
```

界面提供：

- 加载 / 切换模型 bundle
- 单图预测 + 候选区域检测（HSV + 轮廓）
- 视频 / 摄像头实时识别（带帧间跳过 + 置信度平滑）
- 评估 / 错误分析面板

---

## Web 前端（React + TypeScript）

项目已新增浅蓝色现代化 Web 工作台，桌面 PyQt 界面仍保留作为回退方案。Web 版提供：

- 单图分类、Top-5 候选与 HSV 候选区域检测
- 浏览器摄像头 / 本地视频 WebSocket 实时识别
- 最多 50 张图片的批量推理与 CSV 导出
- 当前会话置信度、类别分布、来源和耗时图表
- 模型 Bundle、推理缓存与 43 类标签管理
- 响应式布局、浅蓝 / 深色主题及页面级代码分割

启动后端 API：

```bash
pip install -r requirements.txt
python -m traffic_sign_system.api
# API: http://127.0.0.1:8000
# OpenAPI: http://127.0.0.1:8000/docs
```

启动前端开发服务器（新终端）：

```bash
cd frontend
npm install
npm run dev
# Web UI: http://127.0.0.1:5173
```

生产构建：

```bash
cd frontend
npm run build
npm run preview
```

---

## 候选区域检测（HSV + 轮廓）

`SignDetector` 用 HSV 颜色掩膜 + 几何过滤（面积、宽高比、圆度）在道路场景图中
定位候选交通标志，再送入 `Predictor` 分类。该模块为传统视觉示例，在复杂光照、
遮挡、运动模糊下质量会下降，不用于安全关键场景。

```python
from traffic_sign_system.recognition.sign_detector import SignDetector, draw_detections

detector = SignDetector(predictor)
detections = detector.detect(frame_bgr)
annotated = draw_detections(frame_bgr, detections)
```

---

## 配置说明

所有全局配置集中在 `config/settings.py`：

| 字段 | 含义 | 默认值 |
| --- | --- | --- |
| `IMG_SIZE` | 统一 resize 的方形边长 | `64` |
| `HOG_*` | HOG 窗口 / 块 / cell / 方向数 | `(64,64) (16,16) (8,8) (8,8) 9` |
| `SVM_C / KERNEL / GAMMA` | SVM 超参 | `10.0 / rbf / scale` |
| `TEST_SIZE / VAL_SIZE` | 数据划分比例 | `0.2 / 0.15` |
| `RANDOM_STATE` | 全局随机种子 | `42` |
| `MODEL_ARTIFACTS_DIR` | `.joblib` 输出目录 | `models/artifacts/` |

修改 `settings.py` 即可调整整个系统行为。

---

## 测试与烟雾验证

无 GTSRB 数据集时，可用以下合成数据烟雾测试验证整条 pipeline：

```bash
python -c "
# 仅作语法 / 接口校验，不保证精度
from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.video_recognizer import VideoRecognizer
from traffic_sign_system.recognition.camera_recognizer import CameraRecognizer
from traffic_sign_system.models.train_ensemble import EnsembleClassifier
from traffic_sign_system.scripts.benchmark import run
print('all imports OK')
"
```

---

## 常见问题

- **找不到 GTSRB 数据目录**：脚本会依次探测 `dataset/Train`、`dataset/train/Train`；
  使用 `--train-dir` / `--data` 显式指定即可。
- **OpenCV HOGDescriptor 返回 None**：图像尺寸与 `HOG_WIN_SIZE` 不一致，确认
  `feature_config['img_size']` 与 `settings.IMG_SIZE` 一致。
- **KNN `n_neighbors` 过大**：`--n-neighbors` 不能超过训练集大小。
- **SVM 训练耗时过长**：先用 `--max-samples 2000` 做小样本验证；或开启
  `--grid-search` 配合 `--cv-folds 3` 走 CV。
- **缓存命中率低**：视频中相邻帧若差异较大（场景切换），可降低 `skip_frames`
  并提高 `cache_maxsize`。

---

## License

Course design project — for educational use only.


## 帧间跟踪与复杂场景验证（步骤 24–25）

### 连续视频跟踪验证

```powershell
python -m traffic_sign_system.scripts.verify_tracking `
  traffic_sign_system/tests/assets/tracking_validation.mp4 `
  traffic_sign_system/models/artifacts/svm_hog+hsv.joblib `
  --max-frames 60 --adaptive
```

验证视频每 5 帧注入一次局部遮挡，用于模拟分类瞬时跳变。当前样例实测：原始类别跳变率
40.68%，跟踪后 0%，降低 100%；同一标志始终保持 track_id=0，平均跟踪开销约
0.06 ms/帧。

### 低光照自适应验证

低光照测试图：`traffic_sign_system/tests/assets/low_light_confidence_test.png`。

```powershell
python -m traffic_sign_system.scripts.verify_robustness `
  traffic_sign_system/models/artifacts/svm_hog+hsv.joblib
```

该样例亮度约 53，自动触发 `low_light + fog`，CLAHE clip 从 2.0 调整到 4.0。
当前模型实测从错误类别 11（置信度 14.47%）修正为真实类别 23（置信度 34.46%），
置信度绝对提升约 19.99%。另有完整低光场景及标注结果：
`tests/assets/low_light_test.png`、`tests/assets/low_light_detection_result.png`。

### 自动化测试

```powershell
python -m unittest traffic_sign_system.tests.test_tracking_and_scene -v
```
