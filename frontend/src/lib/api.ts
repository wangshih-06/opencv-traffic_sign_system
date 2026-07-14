import type {
  BatchResponse,
  DetectionResponse,
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
  detect: (file: File, bundle?: string | null) => {
    const body = new FormData();
    body.append("image", file);
    return apiFetch<DetectionResponse>(withBundle("/api/detect", bundle), {
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
};

export function streamUrl(bundle?: string | null, skipFrames = 1) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({ skip_frames: String(skipFrames) });
  if (bundle) params.set("bundle", bundle);
  return `${protocol}//${window.location.host}/ws/stream?${params.toString()}`;
}
