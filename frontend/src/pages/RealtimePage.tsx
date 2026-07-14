import { useEffect, useRef, useState } from "react";
import { Camera, CircleStop, Gauge, Play, Radio, Upload, Video, Wifi, WifiOff } from "lucide-react";
import clsx from "clsx";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { streamUrl } from "../lib/api";
import { formatPercent } from "../lib/format";
import type { CacheStats, StreamMessage, TopKItem } from "../lib/types";
import { useAppStore } from "../store/useAppStore";

export function RealtimePage() {
  const [source, setSource] = useState<"camera" | "video">("camera");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [connection, setConnection] = useState<"idle" | "connecting" | "ready" | "error">("idle");
  const [message, setMessage] = useState("等待启动实时识别");
  const [result, setResult] = useState<(TopKItem & { reused?: boolean }) | null>(null);
  const [fps, setFps] = useState(0);
  const [predictMs, setPredictMs] = useState(0);
  const [cache, setCache] = useState<CacheStats | null>(null);
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
  const lastHistoryRef = useRef({ time: 0, classId: -1 });

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
    }
    setRunning(false);
    sendingRef.current = false;
  };

  useEffect(() => cleanup, []);

  const stop = () => {
    cleanup();
    setConnection("idle");
    setMessage("实时识别已停止");
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
    setConnection("connecting");
    setMessage("正在连接视频源与识别服务…");
    setResult(null);
    setFps(0);
    setPredictMs(0);

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
      socket.onopen = () => setMessage("视频源已就绪，正在加载模型…");
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as StreamMessage;
        if (data.type === "ready") {
          setConnection("ready");
          setRunning(true);
          setMessage("实时识别运行中");
          timerRef.current = window.setInterval(sendFrame, 280);
        } else if (data.type === "prediction" && data.result) {
          sendingRef.current = false;
          setResult(data.result);
          setFps(data.fps ?? 0);
          setPredictMs(data.predict_ms ?? 0);
          setCache(data.cache ?? null);
          const now = Date.now();
          if (!data.result.reused && (now - lastHistoryRef.current.time > 2500 || lastHistoryRef.current.classId !== data.result.class_id)) {
            addHistory({
              source,
              filename: source === "camera" ? "浏览器摄像头" : videoFile?.name ?? "本地视频",
              class_id: data.result.class_id,
              class_name: data.result.class_name,
              confidence: data.result.confidence,
              duration_ms: data.predict_ms ?? 0,
            });
            lastHistoryRef.current = { time: now, classId: data.result.class_id };
          }
        } else if (data.type === "error") {
          sendingRef.current = false;
          setConnection("error");
          setMessage(data.message ?? "识别服务返回错误");
        }
      };
      socket.onerror = () => {
        sendingRef.current = false;
        setConnection("error");
        setMessage("WebSocket 连接失败，请确认后端服务已启动");
      };
      socket.onclose = () => {
        sendingRef.current = false;
        if (running) setConnection("idle");
      };
    } catch (error) {
      cleanup();
      setConnection("error");
      setMessage(error instanceof Error ? error.message : "无法打开视频源");
    }
  };

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
              <video ref={videoRef} muted playsInline className={clsx(!running && "video-stage__inactive")} />
              <canvas ref={canvasRef} hidden />
              {!running && (
                <div className="video-stage__placeholder">
                  <div className="live-visual"><span /><span /><Radio size={34} /></div>
                  <strong>{source === "camera" ? "摄像头尚未启动" : "视频尚未播放"}</strong>
                  <p>{source === "camera" ? "授权浏览器访问摄像头后即可开始实时识别" : "选择本地视频文件并点击开始识别"}</p>
                </div>
              )}
              {running && result && (
                <div className="live-result-overlay">
                  <span>实时识别</span><strong>{result.class_name}</strong><b>{formatPercent(result.confidence)}</b>
                </div>
              )}
              <div className={clsx("connection-badge", `connection-badge--${connection}`)}>
                {connection === "ready" ? <Wifi size={14} /> : <WifiOff size={14} />}{message}
              </div>
            </div>
            <div className="stream-controls">
              <div className="stream-setting">
                <label htmlFor="skip">跳帧</label>
                <select id="skip" value={skipFrames} onChange={(event) => setSkipFrames(Number(event.target.value))} disabled={running}>
                  <option value={0}>不跳帧</option><option value={1}>每 2 帧</option><option value={2}>每 3 帧</option><option value={3}>每 4 帧</option>
                </select>
              </div>
              {running ? <Button variant="danger" icon={<CircleStop size={18} />} onClick={stop}>停止识别</Button> : <Button icon={<Play size={18} />} onClick={start}>开始实时识别</Button>}
            </div>
          </Card>
        </div>

        <aside className="realtime-aside">
          <Card eyebrow="LIVE METRICS" title="实时性能">
            <div className="live-metrics-grid">
              <div><Gauge size={18} /><span>处理帧率</span><strong>{fps.toFixed(1)} <small>FPS</small></strong></div>
              <div><Radio size={18} /><span>推理延迟</span><strong>{predictMs.toFixed(0)} <small>ms</small></strong></div>
              <div><Wifi size={18} /><span>缓存命中率</span><strong>{cache ? `${(cache.hit_rate * 100).toFixed(0)}%` : "—"}</strong></div>
              <div><Camera size={18} /><span>处理帧数</span><strong>{cache?.total ?? 0}</strong></div>
            </div>
          </Card>
          <Card eyebrow="CURRENT RESULT" title="当前识别">
            {result ? (
              <div className="live-current-result">
                <div className="live-current-result__sign">{String(result.class_id).padStart(2, "0")}</div>
                <span>当前最可能类别</span><h3>{result.class_name}</h3>
                <div className="confidence-line"><span style={{ width: `${(result.confidence ?? 0) * 100}%` }} /></div>
                <strong>{formatPercent(result.confidence)} 置信度</strong>
              </div>
            ) : <div className="result-empty result-empty--small"><Radio size={27} /><strong>等待视频帧</strong><p>实时结果将在识别服务就绪后显示。</p></div>}
          </Card>
          <Card eyebrow="TIPS" title="使用建议">
            <ul className="tips-list"><li>保持标志位于画面中央并避免强烈反光</li><li>摄像头识别建议使用 720p 或更高分辨率</li><li>提高跳帧数可以降低后端推理压力</li></ul>
          </Card>
        </aside>
      </div>
    </div>
  );
}
