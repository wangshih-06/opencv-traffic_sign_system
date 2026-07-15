import type {
  BatchResponse,
  DetectionEngine,
  DetectionEnginesResponse,
  DetectionResponse,
  FeedbackCreate,
  FeedbackListResponse,
  FeedbackRecord,
  FeedbackStats,
  FeedbackStatus,
  HealthResponse,
  LabelItem,
  ModelsResponse,
  Prediction,
} from "./types";

async function apiFetch<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: string; message?: string };
      message = payload.detail ?? payload.message ?? message;
    } catch {
      // Keep the HTTP fallback when the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function withBundle(path: string, bundle?: string | null, extra?: Record<string, string>) {
  const params = new URLSearchParams(extra);
  if (bundle) params.set("bundle", bundle);
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

export const api = {
  health: () => apiFetch<HealthResponse>("/api/health"),
  models: () => apiFetch<ModelsResponse>("/api/models"),
  detectionEngines: () => apiFetch<DetectionEnginesResponse>("/api/detection-engines"),
  labels: async () => {
    const response = await apiFetch<{ count: number; items: LabelItem[] }>("/api/labels");
    return response.items;
  },
  loadModel: (name: string) =>
    apiFetch<{ ok: boolean; model: unknown; summary: Record<string, unknown> }>("/api/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  clearCache: (bundle?: string | null) =>
    apiFetch<{ ok: boolean }>(withBundle("/api/models/cache", bundle), { method: "DELETE" }),
  predict: (file: File, bundle?: string | null, topK = 5) => {
    const body = new FormData();
    body.append("image", file);
    return apiFetch<Prediction>(
      withBundle("/api/predict", bundle, { top_k: String(topK) }),
      { method: "POST", body },
    );
  },
  detect: (
    file: File,
    bundle?: string | null,
    engine: DetectionEngine = "traditional",
    detectorModel?: string | null,
  ) => {
    const body = new FormData();
    body.append("image", file);
    const extra: Record<string, string> = { engine };
    if (detectorModel) extra.detector_model = detectorModel;
    return apiFetch<DetectionResponse>(withBundle("/api/detect", bundle, extra), {
      method: "POST",
      body,
    });
  },
  batch: (files: File[], bundle?: string | null) => {
    const body = new FormData();
    files.forEach((file) => body.append("images", file));
    return apiFetch<BatchResponse>(withBundle("/api/batch", bundle), {
      method: "POST",
      body,
    });
  },
  feedback: (status?: FeedbackStatus | "all") => {
    const params = new URLSearchParams();
    if (status && status !== "all") params.set("status", status);
    const query = params.toString();
    return apiFetch<FeedbackListResponse>(`/api/feedback${query ? `?${query}` : ""}`);
  },
  createFeedback: (payload: FeedbackCreate) =>
    apiFetch<{ ok: boolean; item: FeedbackRecord; stats: FeedbackStats }>("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateFeedbackStatus: (id: string, status: FeedbackStatus) =>
    apiFetch<{ ok: boolean; item: FeedbackRecord; stats: FeedbackStats }>(`/api/feedback/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  deleteFeedback: (id: string) =>
    apiFetch<{ ok: boolean; stats: FeedbackStats }>(`/api/feedback/${id}`, { method: "DELETE" }),
};

export function streamUrl(
  bundle?: string | null,
  skipFrames = 1,
  engine: DetectionEngine = "traditional",
  detectorModel?: string | null,
) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({
    skip_frames: String(skipFrames),
    engine,
  });
  if (bundle) params.set("bundle", bundle);
  if (detectorModel) params.set("detector_model", detectorModel);
  return `${protocol}//${window.location.host}/ws/stream?${params.toString()}`;
}
