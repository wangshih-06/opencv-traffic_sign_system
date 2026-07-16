import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DetectionEngine, DetectionResponse, HistoryItem, Prediction } from "../lib/types";

interface AppState {
  selectedModel: string | null;
  detectionEngine: DetectionEngine;
  reviewConfidenceThreshold: number;
  theme: "light" | "dark";
  history: HistoryItem[];
  prediction: Prediction | null;
  detection: DetectionResponse | null;
  setSelectedModel: (model: string | null) => void;
  setDetectionEngine: (engine: DetectionEngine) => void;
  setReviewConfidenceThreshold: (threshold: number) => void;
  toggleTheme: () => void;
  addHistory: (item: Omit<HistoryItem, "id" | "timestamp">) => HistoryItem;
  clearHistory: () => void;
  setPrediction: (prediction: Prediction | null) => void;
  setDetection: (detection: DetectionResponse | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      selectedModel: null,
      detectionEngine: "traditional",
      reviewConfidenceThreshold: 0.7,
      theme: "light",
      history: [],
      prediction: null,
      detection: null,
      setSelectedModel: (selectedModel) => set({ selectedModel }),
      setDetectionEngine: (detectionEngine) => set({ detectionEngine }),
      setReviewConfidenceThreshold: (reviewConfidenceThreshold) => set({
        reviewConfidenceThreshold: Math.min(0.99, Math.max(0.01, reviewConfidenceThreshold)),
      }),
      toggleTheme: () => set((state) => ({ theme: state.theme === "light" ? "dark" : "light" })),
      addHistory: (item) => {
        const entry = { ...item, id: crypto.randomUUID(), timestamp: Date.now() };
        set((state) => ({ history: [entry, ...state.history].slice(0, 60) }));
        return entry;
      },
      clearHistory: () => set({ history: [] }),
      setPrediction: (prediction) => set({ prediction }),
      setDetection: (detection) => set({ detection }),
    }),
    {
      name: "traffic-sign-web-state",
      partialize: (state) => ({
        selectedModel: state.selectedModel,
        detectionEngine: state.detectionEngine,
        reviewConfidenceThreshold: state.reviewConfidenceThreshold,
        theme: state.theme,
        history: state.history,
      }),
    },
  ),
);
