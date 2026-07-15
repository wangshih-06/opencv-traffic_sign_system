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

export type DetectionEngine = "traditional" | "deep" | "hybrid";

export interface DetectionEngineInfo {
  id: DetectionEngine;
  label: string;
  description: string;
  available: boolean;
  degraded?: boolean;
  requires_model: boolean;
}

export interface DeepDetectorModel {
  name: string;
  size_bytes: number;
  modified_at: number;
  metadata: boolean;
}

export interface DetectionEnginesResponse {
  default_engine: DetectionEngine;
  engines: DetectionEngineInfo[];
  deep_models: DeepDetectorModel[];
}

export interface Detection {
  class_id: number;
  class_name: string;
  bbox: [number, number, number, number];
  colour: "red" | "blue";
  confidence: number | null;
  track_id?: number;
  lost_count?: number;
  shape_match?: boolean;
  engine?: DetectionEngine;
  sources?: DetectionEngine[];
  detector_confidence?: number | null;
  deep_class_id?: number;
  deep_class_name?: string;
}

export interface DetectionResponse {
  model: string;
  filename: string;
  detections: Detection[];
  count: number;
  detect_seconds: number;
  cache: Partial<CacheStats>;
  scene?: SceneQuality;
  engine_requested?: DetectionEngine;
  engine_used?: DetectionEngine;
  deep_model?: string | null;
  fallback?: boolean;
  warning?: string | null;
  deep_inference_ms?: number | null;
  image: { width: number; height: number };
}

export interface BatchItemError {
  code: "invalid_image" | "file_too_large" | "unsupported_media_type" | "file_error" | "inference_error";
  message: string;
  status_code: number;
}

export interface BatchSuccessItem {
  ok: true;
  model: string;
  filename: string;
  class_id: number;
  class_name: string;
  confidence: number | null;
  predict_seconds: number;
  top_k: TopKItem[];
  cache: Partial<CacheStats>;
  image: { width: number; height: number };
  error: null;
}

export interface BatchFailureItem {
  ok: false;
  model: null;
  filename: string;
  class_id: null;
  class_name: null;
  confidence: null;
  predict_seconds: 0;
  top_k: [];
  cache: Partial<CacheStats>;
  image: null;
  error: BatchItemError;
}

export type BatchItem = BatchSuccessItem | BatchFailureItem;

export interface BatchResponse {
  model: string;
  count: number;
  success_count: number;
  failed_count: number;
  predict_seconds: number;
  items: BatchItem[];
  cache: Partial<CacheStats>;
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
  model?: string | null;
  class_id: number;
  class_name: string;
  confidence: number | null;
  duration_ms: number;
}


export type FeedbackVerdict = "correct" | "incorrect";
export type FeedbackStatus = "new" | "reviewed" | "exported";

export interface FeedbackCreate {
  history_id?: string;
  source: HistoryItem["source"];
  filename: string;
  model?: string | null;
  predicted_class_id: number;
  predicted_class_name: string;
  predicted_confidence: number | null;
  corrected_class_id?: number | null;
  corrected_class_name?: string | null;
  verdict: FeedbackVerdict;
  note?: string;
  bbox?: [number, number, number, number];
}

export interface FeedbackRecord {
  id: string;
  created_at: string;
  updated_at: string;
  history_id: string | null;
  source: HistoryItem["source"];
  filename: string;
  model: string | null;
  predicted_class_id: number;
  predicted_class_name: string;
  predicted_confidence: number | null;
  corrected_class_id: number;
  corrected_class_name: string;
  verdict: FeedbackVerdict;
  note: string;
  bbox: [number, number, number, number] | null;
  status: FeedbackStatus;
}

export interface FeedbackStats {
  total: number;
  correct: number;
  incorrect: number;
  new: number;
  reviewed: number;
  exported: number;
}

export interface FeedbackListResponse {
  items: FeedbackRecord[];
  count: number;
  stats: FeedbackStats;
}

export interface SceneQuality {
  brightness: number;
  contrast: number;
  blur_score: number;
  noise_score: number;
  degradations: Array<"low_light" | "fog" | "blur" | "noise">;
  quality_score: number;
  quality_status: "good" | "fair" | "poor";
  quality_components: {
    brightness: number;
    contrast: number;
    sharpness: number;
    noise: number;
  };
  analysis_seconds: number;
  recommendations?: Record<string, boolean | number | string>;
}

export interface StreamMessage {
  type: "ready" | "prediction" | "error" | "pong";
  mode?: "detect-track";
  model?: string;
  engine_requested?: DetectionEngine;
  engine_used?: DetectionEngine;
  deep_model?: string | null;
  fallback?: boolean;
  warning?: string | null;
  deep_inference_ms?: number | null;
  message?: string;
  frame_index?: number;
  processed_frames?: number;
  result?: {
    class_id: number;
    class_name: string;
    confidence: number | null;
    bbox?: [number, number, number, number];
    colour?: "red" | "blue";
    track_id?: number;
    lost_count?: number;
    reused?: boolean;
  } | null;
  detections?: Detection[];
  detection_count?: number;
  tracked_count?: number;
  reused?: boolean;
  predict_ms?: number;
  tracker_ms?: number;
  fps?: number;
  cache?: CacheStats;
  scene?: SceneQuality;
  scene_reused?: boolean;
  image?: { width: number; height: number };
}


