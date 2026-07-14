import { useEffect, useState } from "react";
import clsx from "clsx";
import type { Detection } from "../lib/types";

interface DetectionCanvasProps {
  src: string | null;
  detections?: Detection[];
  sourceSize?: { width: number; height: number } | null;
  alt?: string;
}

export function DetectionCanvas({ src, detections = [], sourceSize, alt = "待识别图片" }: DetectionCanvasProps) {
  const [loadedSize, setLoadedSize] = useState<{ width: number; height: number } | null>(null);
  useEffect(() => setLoadedSize(null), [src]);
  const size = sourceSize ?? loadedSize;

  if (!src) {
    return (
      <div className="image-stage image-stage--empty">
        <div className="stage-orbit stage-orbit--one" />
        <div className="stage-orbit stage-orbit--two" />
        <div className="stage-placeholder-mark">AI</div>
        <strong>图像预览区</strong>
        <span>选择图片后将在此处显示</span>
      </div>
    );
  }

  return (
    <div className="image-stage">
      <div className="image-stage__frame">
        <img
          src={src}
          alt={alt}
          onLoad={(event) => setLoadedSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
        />
        {size && detections.map((detection, index) => {
          const [x, y, width, height] = detection.bbox;
          return (
            <div
              className={clsx("detection-box", `detection-box--${detection.colour}`)}
              key={`${x}-${y}-${index}`}
              style={{
                left: `${(x / size.width) * 100}%`,
                top: `${(y / size.height) * 100}%`,
                width: `${(width / size.width) * 100}%`,
                height: `${(height / size.height) * 100}%`,
              }}
            >
              <span>{detection.class_name} · {detection.confidence == null ? "—" : `${Math.round(detection.confidence * 100)}%`}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
