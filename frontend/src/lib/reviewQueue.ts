import { api } from "./api";
import type { HistoryItem } from "./types";

/**
 * Persist only actionable low-confidence results. The API deduplicates by
 * history ID, so retries and React development re-renders cannot create a
 * second task for the same recognition result.
 */
export function enqueueLowConfidenceReview(item: HistoryItem, threshold: number) {
  if (item.confidence == null || item.confidence >= threshold) return;
  void api.enqueueReview({
    history_id: item.id,
    source: item.source,
    filename: item.filename,
    model: item.model,
    predicted_class_id: item.class_id,
    predicted_class_name: item.class_name,
    predicted_confidence: item.confidence,
    reason: "low_confidence",
  }).catch(() => {
    // Recognition must remain usable if the optional review service is offline.
  });
}
