import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { BrainCircuit, Cpu, Layers3 } from "lucide-react";
import clsx from "clsx";
import { api } from "../lib/api";
import type { DetectionEngine } from "../lib/types";
import { useAppStore } from "../store/useAppStore";

const ICONS = {
  traditional: Cpu,
  deep: BrainCircuit,
  hybrid: Layers3,
};

export function DetectionEngineControl({
  disabled = false,
  compact = false,
}: {
  disabled?: boolean;
  compact?: boolean;
}) {
  const engine = useAppStore((state) => state.detectionEngine);
  const setEngine = useAppStore((state) => state.setDetectionEngine);
  const query = useQuery({
    queryKey: ["detection-engines"],
    queryFn: api.detectionEngines,
    staleTime: 30_000,
  });
  const engines = query.data?.engines ?? [
    {
      id: "traditional" as const,
      label: "传统引擎",
      description: "HSV/轮廓候选区域 + HOG/HSV 分类器",
      available: true,
      requires_model: true,
    },
  ];
  const current = engines.find((item) => item.id === engine) ?? engines[0];
  const CurrentIcon = ICONS[current.id];

  useEffect(() => {
    if (query.data && !query.data.engines.find((item) => item.id === engine)?.available) {
      setEngine(query.data.default_engine);
    }
  }, [engine, query.data, setEngine]);

  return (
    <div className={clsx("engine-control", compact && "engine-control--compact")}>
      <div className="engine-control__heading">
        <span className={clsx("engine-control__icon", `engine-control__icon--${current.id}`)}>
          <CurrentIcon size={16} />
        </span>
        <div>
          <strong>检测引擎</strong>
          <small>{current.description}</small>
        </div>
      </div>
      <select
        value={engine}
        disabled={disabled || query.isLoading}
        onChange={(event) => setEngine(event.target.value as DetectionEngine)}
        aria-label="选择检测引擎"
      >
        {engines.map((item) => (
          <option key={item.id} value={item.id} disabled={!item.available}>
            {item.label}{!item.available ? "（未安装模型）" : item.degraded ? "（传统降级）" : ""}
          </option>
        ))}
      </select>
      {!compact && (
        <div className="engine-control__status">
          <span className={clsx(current.available ? "ready" : "missing")}>
            {current.available ? "可用" : "不可用"}
          </span>
          <small>
            {query.data?.deep_models.length
              ? `ONNX：${query.data.deep_models[0].name}`
              : "放入 ONNX 检测模型后可启用深度与完整混合模式"}
          </small>
        </div>
      )}
      {query.error && <small className="engine-control__error">无法读取引擎状态</small>}
    </div>
  );
}
