import clsx from "clsx";
import { CheckCircle2, Gauge, Timer, Trophy } from "lucide-react";
import type { Prediction, TopKItem } from "../lib/types";
import { confidenceTone, formatDuration, formatPercent } from "../lib/format";

function ConfidenceRing({ value }: { value: number | null }) {
  const percent = value == null ? 0 : Math.round(value * 100);
  return (
    <div
      className={clsx("confidence-ring", `confidence-ring--${confidenceTone(value)}`)}
      style={{ "--confidence": `${percent * 3.6}deg` } as React.CSSProperties}
      aria-label={`置信度 ${formatPercent(value)}`}
    >
      <div>
        <strong>{value == null ? "—" : `${percent}%`}</strong>
        <span>置信度</span>
      </div>
    </div>
  );
}

export function TopKList({ items }: { items: TopKItem[] }) {
  if (!items.length) {
    return <div className="inline-empty">当前模型未提供概率分布</div>;
  }
  const max = Math.max(...items.map((item) => item.confidence), 0.01);
  return (
    <div className="top-k-list">
      {items.map((item, index) => (
        <div className="top-k-item" key={item.class_id}>
          <span className={clsx("top-k-item__rank", index < 3 && `rank-${index + 1}`)}>{index + 1}</span>
          <div className="top-k-item__content">
            <div className="top-k-item__label">
              <span>{item.class_name}</span>
              <strong>{formatPercent(item.confidence)}</strong>
            </div>
            <div className="progress-track">
              <span style={{ width: `${(item.confidence / max) * 100}%` }} />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function PredictionPanel({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) {
    return (
      <div className="result-empty">
        <div className="result-empty__icon"><Gauge size={30} /></div>
        <strong>等待识别结果</strong>
        <p>上传交通标志图片并开始识别，结果与 Top-5 候选会显示在这里。</p>
      </div>
    );
  }

  return (
    <div className="prediction-panel">
      <div className="prediction-hero">
        <ConfidenceRing value={prediction.confidence} />
        <div className="prediction-hero__copy">
          <span className="success-label"><CheckCircle2 size={15} /> 识别完成</span>
          <h3>{prediction.class_name}</h3>
          <p>类别编号 #{String(prediction.class_id).padStart(2, "0")}</p>
        </div>
      </div>
      <div className="metric-grid metric-grid--compact">
        <div className="metric-chip"><Timer size={17} /><span>推理耗时</span><strong>{formatDuration(prediction.predict_seconds)}</strong></div>
        <div className="metric-chip"><Trophy size={17} /><span>Top-1</span><strong>{formatPercent(prediction.confidence)}</strong></div>
      </div>
      <div className="section-title"><span>候选类别</span><small>Top {prediction.top_k.length}</small></div>
      <TopKList items={prediction.top_k} />
    </div>
  );
}
