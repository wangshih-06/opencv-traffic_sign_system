import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DetectionResponse, HistoryItem, Prediction } from "../lib/types";

interface AppState {
  selectedModel: string | null;
  theme: "light" | "dark";
  history: HistoryItem[];
  prediction: Prediction | null;
  detection: DetectionResponse | null;
  setSelectedModel: (model: string | null) => void;
  toggleTheme: () => void;
  addHistory: (item: Omit<HistoryItem, "id" | "timestamp">) => void;
  clearHistory: () => void;
  setPrediction: (prediction: Prediction | null) => void;
  setDetection: (detection: DetectionResponse | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      selectedModel: null,
      theme: "light",
      history: [],
      prediction: null,
      detection: null,
      setSelectedModel: (selectedModel) => set({ selectedModel }),
      toggleTheme: () => set((state) => ({ theme: state.theme === "light" ? "dark" : "light" })),
      addHistory: (item) =>
        set((state) => ({
          history: [
            { ...item, id: crypto.randomUUID(), timestamp: Date.now() },
            ...state.history,
          ].slice(0, 60),
        })),
      clearHistory: () => set({ history: [] }),
      setPrediction: (prediction) => set({ prediction }),
      setDetection: (detection) => set({ detection }),
    }),
    {
      name: "traffic-sign-web-state",
      partialize: (state) => ({
        selectedModel: state.selectedModel,
        theme: state.theme,
        history: state.history,
      }),
    },
  ),
);
