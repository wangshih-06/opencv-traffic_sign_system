# ONNX 深度检测模型

将 OpenCV DNN 可加载的 ONNX 目标检测模型放入此目录，Web 前端即可在引擎选择器中启用深度或混合引擎。

默认文件名：

```text
traffic_sign_detector.onnx
```

可选的同名 JSON sidecar 用于声明输入尺寸、类别标签和阈值：

```json
{
  "input_size": 640,
  "num_classes": 43,
  "labels": {
    "0": "限速20公里/小时",
    "1": "限速30公里/小时"
  },
  "confidence_threshold": 0.35,
  "nms_threshold": 0.45,
  "output_format": "auto"
}
```

当前解码器支持：

- YOLOv5 原始输出：`N x (5 + classes)`
- YOLOv8 原始输出：`N x (4 + classes)`，包括常见转置形式
- NMS-ready 输出：`N x 6`，格式为 `x1,y1,x2,y2,score,class_id`
- OpenCV/SSD 输出：`N x 7`

ONNX 二进制模型默认被 `.gitignore` 忽略，请通过制品仓库或发布包分发。
