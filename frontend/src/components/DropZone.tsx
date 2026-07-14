import { useCallback, useRef, useState } from "react";
import clsx from "clsx";
import { FileImage, UploadCloud } from "lucide-react";
import { Button } from "./Button";

interface DropZoneProps {
  files?: File[];
  multiple?: boolean;
  compact?: boolean;
  onFiles: (files: File[]) => void;
  accept?: string;
}

export function DropZone({
  files = [],
  multiple = false,
  compact = false,
  onFiles,
  accept = "image/jpeg,image/png,image/webp,image/bmp",
}: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const select = useCallback(
    (list: FileList | null) => {
      if (!list) return;
      const selected = Array.from(list);
      onFiles(multiple ? selected : selected.slice(0, 1));
    },
    [multiple, onFiles],
  );

  return (
    <div
      className={clsx("drop-zone", dragging && "drop-zone--active", compact && "drop-zone--compact")}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        event.preventDefault();
        if (event.currentTarget === event.target) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        select(event.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
      }}
      aria-label="上传图片"
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        hidden
        onChange={(event) => select(event.target.files)}
      />
      <div className="drop-zone__icon">
        {files.length ? <FileImage size={28} /> : <UploadCloud size={29} />}
      </div>
      <div className="drop-zone__copy">
        <strong>{files.length ? `已选择 ${files.length} 个文件` : "拖拽图片到这里"}</strong>
        <span>
          {files.length
            ? files.length === 1
              ? files[0].name
              : "可继续拖入文件以替换当前选择"
            : `支持 JPG、PNG、WebP、BMP${multiple ? "，单次最多 50 张" : "，不超过 20 MB"}`}
        </span>
      </div>
      <Button type="button" variant="secondary" size="sm" onClick={(event) => event.stopPropagation()}>
        浏览文件
      </Button>
    </div>
  );
}
