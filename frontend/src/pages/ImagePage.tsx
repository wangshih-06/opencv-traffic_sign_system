import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Crosshair, Eraser, History, ImageIcon, ScanSearch, Sparkles, Timer, Zap } from "lucide-react";
import clsx from "clsx";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { DetectionCanvas } from "../components/DetectionCanvas";
import { DetectionEngineControl } from "../components/DetectionEngineControl";
import { DropZone } from "../components/DropZone";
import { PredictionPanel } from "../components/PredictionPanel";
import { api } from "../lib/api";
import { formatDuration, formatPercent, formatTime } from "../lib/format";
import { useAppStore } from "../store/useAppStore";

export function ImagePage() {
  const [mode, setMode] = useState<"classify" | "detect">("classify");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const selectedModel = useAppStore((state) => state.selectedModel);
  const detectionEngine = useAppStore((state) => state.detectionEngine);
  const prediction = useAppStore((state) => state.prediction);
  const detection = useAppStore((state) => state.detection);
  const setPrediction = useAppStore((state) => state.setPrediction);
  const setDetection = useAppStore((state) => state.setDetection);
  const addHistory = useAppStore((state) => state.addHistory);
  const history = useAppStore((state) => state.history);
  const clearHistory = useAppStore((state) => state.clearHistory);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const classifyMutation = useMutation({
    mutationFn: () => api.predict(file!, selectedModel, 5),
    onSuccess: (result) => {
      setPrediction(result);
      setDetection(null);
      addHistory({
        source: "image",
        filename: file?.name ?? "image",
        model: selectedModel,
        class_id: result.class_id,
        class_name: result.class_name,
        confidence: result.confidence,
        duration_ms: result.predict_seconds * 1000,
      });
    },
  });

  const detectMutation = useMutation({
    mutationFn: () => api.detect(file!, selectedModel, detectionEngine),
    onSuccess: (result) => {
      setDetection(result);
      setPrediction(null);
      result.detections.forEach((item) => addHistory({
        source: "image",
        filename: file?.name ?? "image",
        model: selectedModel,
        class_id: item.class_id,
        class_name: item.class_name,
        confidence: item.confidence,
        duration_ms: result.detect_seconds * 1000,
      }));
    },
  });

  const activeMutation = mode === "classify" ? classifyMutation : detectMutation;
  const error = activeMutation.error instanceof Error ? activeMutation.error.message : null;

  const handleFile = (files: File[]) => {
    const next = files[0] ?? null;
    if (preview) URL.revokeObjectURL(preview);
    setFile(next);
    setPreview(next ? URL.createObjectURL(next) : null);
    setPrediction(null);
    setDetection(null);
    classifyMutation.reset();
    detectMutation.reset();
  };

  const recent = useMemo(() => history.slice(0, 5), [history]);

  return (
    <div className="page-stack">
      <div className="quick-stats">
        <div className="quick-stat quick-stat--blue"><span><Zap size={18} /></span><div><strong>43</strong><small>支持标志类别</small></div></div>
        <div className="quick-stat quick-stat--cyan"><span><Sparkles size={18} /></span><div><strong>HOG + HSV</strong><small>融合特征模式</small></div></div>
        <div className="quick-stat quick-stat--violet"><span><Timer size={18} /></span><div><strong>{prediction ? formatDuration(prediction.predict_seconds) : "实时"}</strong><small>当前推理速度</small></div></div>
      </div>

      <div className="workspace-grid">
        <div className="workspace-main">
          <Card className="mode-card">
            <div className="segmented-control" role="tablist" aria-label="识别模式">
              <button className={clsx(mode === "classify" && "active")} onClick={() => setMode("classify")} role="tab" aria-selected={mode === "classify"}>
                <ImageIcon size={17} /><span><strong>分类识别</strong><small>识别单个标志类别</small></span>
              </button>
              <button className={clsx(mode === "detect" && "active")} onClick={() => setMode("detect")} role="tab" aria-selected={mode === "detect"}>
                <ScanSearch size={17} /><span><strong>场景检测</strong><small>查找并识别多个标志</small></span>
              </button>
            </div>
          </Card>

          {mode === "detect" && (
            <Card eyebrow="DETECTION ENGINE" title="选择识别后端">
              <DetectionEngineControl />
            </Card>
          )}

          <Card
            eyebrow="INPUT IMAGE"
            title="上传识别图片"
            action={file && <Button variant="ghost" size="sm" icon={<Eraser size={15} />} onClick={() => handleFile([])}>清除</Button>}
          >
            <DropZone files={file ? [file] : []} onFiles={handleFile} />
          </Card>

          <Card className="preview-card" eyebrow="VISION CANVAS" title={mode === "classify" ? "图像预览" : "候选区域检测"}>
            <DetectionCanvas
              src={preview}
              detections={mode === "detect" ? detection?.detections : []}
              sourceSize={mode === "detect" ? detection?.image : null}
              alt={file?.name}
            />
            <div className="preview-toolbar">
              <div className="preview-toolbar__hint"><Crosshair size={16} />{file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : "选择清晰、标志主体完整的图片可获得更佳结果"}</div>
              <Button
                size="lg"
                icon={mode === "classify" ? <Sparkles size={18} /> : <ScanSearch size={18} />}
                disabled={!file || !selectedModel}
                loading={activeMutation.isPending}
                onClick={() => activeMutation.mutate()}
              >
                {mode === "classify" ? "开始智能识别" : "开始场景检测"}
              </Button>
            </div>
            {error && <div className="error-message">{error}</div>}
          </Card>
        </div>

        <aside className="workspace-aside">
          <Card eyebrow="AI RESULT" title={mode === "classify" ? "识别结果" : "检测结果"}>
            {mode === "classify" ? (
              <PredictionPanel prediction={prediction} />
            ) : detection ? (
              <div className="detection-results">
                <div className="detection-summary"><div><strong>{detection.count}</strong><span>检测到的标志</span></div><div><strong>{formatDuration(detection.detect_seconds)}</strong><span>总处理耗时</span></div></div>
                <div className={clsx("engine-result-badge", detection.fallback && "engine-result-badge--warning")}>
                  <strong>
                    {detection.engine_used === "deep"
                      ? "深度引擎"
                      : detection.engine_used === "hybrid"
                        ? "混合引擎"
                        : "传统引擎"}
                    {detection.fallback ? " · 已回退" : ""}
                  </strong>
                  {detection.deep_inference_ms != null && <span>ONNX {detection.deep_inference_ms.toFixed(1)} ms</span>}
                  {detection.warning && <small>{detection.warning}</small>}
                </div>
                {detection.detections.length ? detection.detections.map((item, index) => (
                  <div className="detection-result-item" key={`${item.class_id}-${index}`}>
                    <span className={clsx("colour-dot", `colour-dot--${item.colour}`)} />
                    <div><strong>{item.class_name}</strong><small>类别 #{item.class_id} · {item.colour === "red" ? "红色候选" : "蓝色候选"}</small></div>
                    <b>{formatPercent(item.confidence)}</b>
                  </div>
                )) : <div className="result-empty result-empty--small"><ScanSearch size={27} /><strong>未检测到候选标志</strong><p>可尝试更换主体更清晰的道路场景图片。</p></div>}
              </div>
            ) : (
              <div className="result-empty"><ScanSearch size={30} /><strong>等待检测</strong><p>系统会使用 HSV 与轮廓筛选定位红色、蓝色交通标志。</p></div>
            )}
          </Card>

          <Card
            eyebrow="RECENT ACTIVITY"
            title="最近识别"
            action={history.length > 0 && <button className="text-button" onClick={clearHistory}>清空</button>}
          >
            {recent.length ? (
              <div className="history-list">
                {recent.map((item) => (
                  <div className="history-item" key={item.id}>
                    <div className="history-item__icon"><History size={16} /></div>
                    <div><strong>{item.class_name}</strong><small>{item.filename} · {formatTime(item.timestamp)}</small></div>
                    <span>{formatPercent(item.confidence, 0)}</span>
                  </div>
                ))}
              </div>
            ) : <div className="inline-empty">暂无识别记录</div>}
          </Card>
        </aside>
      </div>
    </div>
  );
}

