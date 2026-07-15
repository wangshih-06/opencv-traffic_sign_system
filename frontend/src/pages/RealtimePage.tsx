import { useEffect, useRef, useState } from "react";
import {
  Activity,
  Camera,
  CircleStop,
  Crosshair,
  Gauge,
  Layers3,
  Play,
  Radio,
  ScanSearch,
  Upload,
  Video,
  Wifi,
  WifiOff,
} from "lucide-react";
import clsx from "clsx";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { streamUrl } from "../lib/api";
import { formatPercent } from "../lib/format";
import type { Detection, StreamMessage } from "../lib/types";
import { useAppStore } from "../store/useAppStore";

type FrameSize = { width: number; height: number };

function DetectionOverlay({
  detections,
  frameSize,
  selectedTrackId,
  onSelect,
}: {
  detections: Detection[];
  frameSize: FrameSize | null;
  selectedTrackId: number | null;
  onSelect: (trackId: number) => void;
}) {
  if (!frameSize || !detections.length) return null;

  return (
    <svg
      className="tracking-overlay"
      viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="实时交通标志检测框"
    >
      {detections.map((detection, index) => {
        const [x, y, width, height] = detection.bbox;
        const trackId = detection.track_id ?? index;
        const selected = selectedTrackId === trackId;
        const lost = (detection.lost_count ?? 0) > 0;
        const label = `#${trackId} ${detection.class_name} ${formatPercent(detection.confidence)}`;
        const labelWidth = Math.min(
          Math.max(128, label.length * 8 + 18),
          Math.max(128, frameSize.width - x),
        );
        const labelY = Math.max(0, y - 23);

        return (
          <g
            key={`${trackId}-${index}`}
            className={clsx(
              "tracking-overlay__item",
              `tracking-overlay__item--${detection.colour}`,
              lost && "tracking-overlay__item--lost",
              selected && "tracking-overlay__item--selected",
            )}
            onClick={() => onSelect(trackId)}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") onSelect(trackId);
            }}
          >
            <rect className="tracking-overlay__hitbox" x={x} y={y} width={width} height={height} />
            <rect className="tracking-overlay__box" x={x} y={y} width={width} height={height} rx={4} />
            <rect className="tracking-overlay__label-bg" x={x} y={labelY} width={labelWidth} height={22} rx={5} />
            <text className="tracking-overlay__label" x={x + 9} y={labelY + 15}>
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function RealtimePage() {
  const [source, setSource] = useState<"camera" | "video">("camera");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [connection, setConnection] = useState<"idle" | "connecting" | "ready" | "error">("idle");
  const [message, setMessage] = useState("等待启动实时检测");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameSize, setFrameSize] = useState<FrameSize | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [predictMs, setPredictMs] = useState(0);
  const [trackerMs, setTrackerMs] = useState(0);
  const [processedFrames, setProcessedFrames] = useState(0);
  const [skipFrames, setSkipFrames] = useState(1);
  const selectedModel = useAppStore((state) => state.selectedModel);
  const addHistory = useAppStore((state) => state.addHistory);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const mediaRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const sendingRef = useRef(false);
  const historyRef = useRef(new Map<number, number>());

  const cleanup = () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    mediaRef.current?.getTracks().forEach((track) => track.stop());
    mediaRef.current = null;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute("src");
      videoRef.current.load();
    }
    setRunning(false);
    sendingRef.current = false;
  };

  useEffect(() => cleanup, []);

  const stop = () => {
    cleanup();
    setConnection("idle");
    setMessage("实时检测已停止");
  };

  const sendFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const socket = socketRef.current;
    if (!video || !canvas || !socket || socket.readyState !== WebSocket.OPEN || video.readyState < 2 || sendingRef.current) return;

    const maxWidth = 720;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    sendingRef.current = true;
    canvas.toBlob((blob) => {
      if (blob && socket.readyState === WebSocket.OPEN) socket.send(blob);
      else sendingRef.current = false;
    }, "image/jpeg", 0.76);
  };

  const start = async () => {
    if (!selectedModel) {
      setConnection("error");
      setMessage("请先选择可用模型");
      return;
    }
    if (source === "video" && !videoFile) {
      setConnection("error");
      setMessage("请先选择本地视频文件");
      return;
    }

    cleanup();
    historyRef.current.clear();
    setConnection("connecting");
    setMessage("正在连接视频源与多目标检测服务…");
    setDetections([]);
    setFrameSize(null);
    setSelectedTrackId(null);
    setFps(0);
    setPredictMs(0);
    setTrackerMs(0);
    setProcessedFrames(0);

    try {
      const video = videoRef.current!;
      if (source === "camera") {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
          audio: false,
        });
        mediaRef.current = stream;
        video.srcObject = stream;
      } else {
        objectUrlRef.current = URL.createObjectURL(videoFile!);
        video.src = objectUrlRef.current;
        video.loop = true;
      }
      await video.play();

      const socket = new WebSocket(streamUrl(selectedModel, skipFrames));
      socketRef.current = socket;
      socket.onopen = () => setMessage("视频源已就绪，正在加载检测模型…");
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as StreamMessage;
        if (data.type === "ready") {
          setConnection("ready");
          setRunning(true);
          setMessage("实时多目标检测运行中");
          timerRef.current = window.setInterval(sendFrame, 280);
        } else if (data.type === "prediction") {
          sendingRef.current = false;
          const nextDetections = data.detections ?? [];
          setDetections(nextDetections);
          setFrameSize(data.image ?? null);
          setFps(data.fps ?? 0);
          setPredictMs(data.predict_ms ?? 0);
          setTrackerMs(data.tracker_ms ?? 0);
          setProcessedFrames(data.processed_frames ?? 0);
          setSelectedTrackId((current) =>
            current !== null && !nextDetections.some((item) => item.track_id === current)
              ? null
              : current,
          );

          const now = Date.now();
          if (!data.reused) {
            nextDetections
              .filter((item) => (item.lost_count ?? 0) === 0 && item.track_id !== undefined)
              .forEach((item) => {
                const trackId = item.track_id!;
                const lastSeen = historyRef.current.get(trackId) ?? 0;
                if (now - lastSeen < 5000) return;
                addHistory({
                  source,
                  filename: source === "camera" ? "浏览器摄像头" : videoFile?.name ?? "本地视频",
                  model: selectedModel,
                  class_id: item.class_id,
                  class_name: item.class_name,
                  confidence: item.confidence,
                  duration_ms: (data.predict_ms ?? 0) + (data.tracker_ms ?? 0),
                });
                historyRef.current.set(trackId, now);
              });
          }
        } else if (data.type === "error") {
          sendingRef.current = false;
          setConnection("error");
          setMessage(data.message ?? "检测服务返回错误");
        }
      };
      socket.onerror = () => {
        sendingRef.current = false;
        setConnection("error");
        setMessage("WebSocket 连接失败，请确认后端服务已启动");
      };
      socket.onclose = () => {
        sendingRef.current = false;
        setRunning(false);
      };
    } catch (error) {
      cleanup();
      setConnection("error");
      setMessage(error instanceof Error ? error.message : "无法打开视频源");
    }
  };

  const selectedDetection = selectedTrackId === null
    ? detections[0] ?? null
    : detections.find((item) => item.track_id === selectedTrackId) ?? detections[0] ?? null;

  return (
    <div className="page-stack">
      <div className="realtime-layout">
        <div className="realtime-main">
          <Card>
            <div className="source-switch">
              <button className={clsx(source === "camera" && "active")} onClick={() => !running && setSource("camera")} disabled={running}><Camera size={18} /><span><strong>浏览器摄像头</strong><small>使用设备实时画面</small></span></button>
              <button className={clsx(source === "video" && "active")} onClick={() => !running && setSource("video")} disabled={running}><Video size={18} /><span><strong>本地视频</strong><small>上传视频进行识别</small></span></button>
            </div>
            {source === "video" && (
              <label className="video-file-picker">
                <Upload size={17} /><span>{videoFile?.name ?? "选择 MP4、WebM 或 MOV 视频"}</span>
                <input type="file" accept="video/mp4,video/webm,video/quicktime" hidden onChange={(event) => setVideoFile(event.target.files?.[0] ?? null)} disabled={running} />
              </label>
            )}
          </Card>

          <Card className="stream-card" padded={false}>
            <div className="video-stage">
              <div className="video-frame">
                <video ref={videoRef} muted playsInline className={clsx(!running && "video-stage__inactive")} />
                {running && (
                  <DetectionOverlay
                    detections={detections}
                    frameSize={frameSize}
                    selectedTrackId={selectedTrackId}
                    onSelect={setSelectedTrackId}
                  />
                )}
              </div>
              <canvas ref={canvasRef} hidden />
              {!running && (
                <div className="video-stage__placeholder">
                  <div className="live-visual"><span /><span /><Radio size={34} /></div>
                  <strong>{source === "camera" ? "摄像头尚未启动" : "视频尚未播放"}</strong>
                  <p>{source === "camera" ? "授权浏览器访问摄像头后即可开始多目标检测" : "选择本地视频并点击开始检测"}</p>
                </div>
              )}
              {running && (
                <div className="live-result-overlay live-result-overlay--multi">
                  <span><ScanSearch size={12} />实时多目标检测</span>
                  <strong>{detections.length ? `${detections.length} 个目标正在跟踪` : "暂未发现交通标志"}</strong>
                  <b>{selectedDetection ? `当前 #${selectedDetection.track_id ?? "?"}` : "等待目标"}</b>
                </div>
              )}
              <div className={clsx("connection-badge", `connection-badge--${connection}`)}>
                {connection === "ready" ? <Wifi size={14} /> : <WifiOff size={14} />}{message}
              </div>
            </div>
            <div className="stream-controls">
              <div className="stream-setting">
                <label htmlFor="skip">检测跳帧</label>
                <select id="skip" value={skipFrames} onChange={(event) => setSkipFrames(Number(event.target.value))} disabled={running}>
                  <option value={0}>不跳帧</option><option value={1}>每 2 帧</option><option value={2}>每 3 帧</option><option value={3}>每 4 帧</option>
                </select>
              </div>
              {running ? <Button variant="danger" icon={<CircleStop size={18} />} onClick={stop}>停止检测</Button> : <Button icon={<Play size={18} />} onClick={start}>开始多目标检测</Button>}
            </div>
          </Card>
        </div>

        <aside className="realtime-aside">
          <Card eyebrow="LIVE METRICS" title="实时性能">
            <div className="live-metrics-grid">
              <div><Gauge size={18} /><span>检测帧率</span><strong>{fps.toFixed(1)} <small>FPS</small></strong></div>
              <div><Radio size={18} /><span>检测延迟</span><strong>{predictMs.toFixed(0)} <small>ms</small></strong></div>
              <div><Activity size={18} /><span>跟踪耗时</span><strong>{trackerMs.toFixed(2)} <small>ms</small></strong></div>
              <div><Layers3 size={18} /><span>处理帧数</span><strong>{processedFrames}</strong></div>
            </div>
          </Card>

          <Card eyebrow="TRACKED OBJECTS" title={`当前目标 ${detections.length} 个`}>
            {detections.length ? (
              <div className="tracked-list">
                {detections.map((item) => {
                  const trackId = item.track_id ?? -1;
                  const lost = (item.lost_count ?? 0) > 0;
                  return (
                    <button
                      className={clsx("tracked-item", selectedTrackId === trackId && "tracked-item--selected", lost && "tracked-item--lost")}
                      key={trackId}
                      onClick={() => setSelectedTrackId(trackId)}
                    >
                      <span className={clsx("tracked-item__id", `tracked-item__id--${item.colour}`)}>#{trackId}</span>
                      <span className="tracked-item__main"><strong>{item.class_name}</strong><small>{lost ? "短暂丢失 · 继续跟踪" : "当前帧已检测"}</small></span>
                      <b>{formatPercent(item.confidence)}</b>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="result-empty result-empty--small"><Crosshair size={27} /><strong>等待检测目标</strong><p>检测到多个交通标志后会在此显示 Track ID。</p></div>
            )}
          </Card>

          <Card eyebrow="CURRENT TARGET" title="当前选中目标">
            {selectedDetection ? (
              <div className="live-current-result">
                <div className={clsx("live-current-result__sign", `live-current-result__sign--${selectedDetection.colour}`)}>{String(selectedDetection.class_id).padStart(2, "0")}</div>
                <span>Track ID #{selectedDetection.track_id ?? "?"}</span><h3>{selectedDetection.class_name}</h3>
                <div className="confidence-line"><span style={{ width: `${(selectedDetection.confidence ?? 0) * 100}%` }} /></div>
                <strong>{formatPercent(selectedDetection.confidence)} 置信度</strong>
              </div>
            ) : <div className="result-empty result-empty--small"><Radio size={27} /><strong>等待视频帧</strong><p>实时目标将在检测服务就绪后显示。</p></div>}
          </Card>

          <Card eyebrow="TIPS" title="使用建议">
            <ul className="tips-list"><li>保持标志位于画面中央并避免强烈反光</li><li>检测框颜色代表候选标志的主色</li><li>同一目标会保持稳定的 Track ID</li><li>提高跳帧数可以降低后端推理压力</li></ul>
          </Card>
        </aside>
      </div>
    </div>
  );
}
