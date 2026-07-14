export interface CacheStats {
  hits: number;
  misses: number;
  total: number;
  hit_rate: number;
  size: number;
  maxsize: number;
}

export interface TopKItem {
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface Prediction extends TopKItem {
  model?: string;
  filename?: string;
  predict_seconds: number;
  top_k: TopKItem[];
  cache: CacheStats;
  image: { width: number; height: number };
  reused?: boolean;
}

export interface Detection {
  class_id: number;
  class_name: string;
  bbox: [number, number, number, number];
  colour: "red" | "blue";
  confidence: number | null;
}

export interface DetectionResponse {
  model: string;
  filename: string;
  detections: Detection[];
  count: number;
  detect_seconds: number;
  cache: CacheStats;
  image: { width: number; height: number };
}

export interface BatchItem {
  class_id: number;
  class_name: string;
  filename: string;
  confidence: number | null;
}

export interface BatchResponse {
  model: string;
  count: number;
  predict_seconds: number;
  items: BatchItem[];
  cache: CacheStats;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_available: boolean;
  active_model: string | null;
  loaded_models: number;
  labels: number;
}

export interface ModelBundle {
  name: string;
  size_bytes: number;
  modified_at: number;
  loaded: boolean;
  active: boolean;
  classifier: string | null;
  feature_mode: string | null;
  feature_dim: number | null;
  cache: CacheStats | null;
}

export interface ModelsResponse {
  active_model: string | null;
  default_model: string;
  bundles: ModelBundle[];
}

export interface LabelItem {
  class_id: number;
  class_name: string;
}

export interface HistoryItem {
  id: string;
  timestamp: number;
  source: "image" | "camera" | "video" | "batch";
  filename: string;
  class_id: number;
  class_name: string;
  confidence: number | null;
  duration_ms: number;
}

export interface StreamMessage {
  type: "ready" | "prediction" | "error" | "pong";
  model?: string;
  message?: string;
  frame_index?: number;
  result?: TopKItem & { reused?: boolean };
  predict_ms?: number;
  fps?: number;
  cache?: CacheStats;
}


