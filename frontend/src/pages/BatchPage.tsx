import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, Download, Files, ListChecks, Play, Trash2, XCircle } from "lucide-react";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { DropZone } from "../components/DropZone";
import { api } from "../lib/api";
import { formatDuration, formatPercent } from "../lib/format";
import type { BatchResponse } from "../lib/types";
import { useAppStore } from "../store/useAppStore";

export function BatchPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const selectedModel = useAppStore((state) => state.selectedModel);
  const addHistory = useAppStore((state) => state.addHistory);

  const mutation = useMutation({
    mutationFn: () => api.batch(files, selectedModel),
    onSuccess: (response) => {
      setResult(response);
      response.items.forEach((item) => addHistory({
        source: "batch",
        filename: item.filename,
        model: selectedModel,
        class_id: item.class_id,
        class_name: item.class_name,
        confidence: item.confidence,
        duration_ms: response.predict_seconds * 1000 / Math.max(response.count, 1),
      }));
    },
  });

  const summary = useMemo(() => {
    if (!result) return [];
    const counts = new Map<string, number>();
    result.items.forEach((item) => counts.set(item.class_name, (counts.get(item.class_name) ?? 0) + 1));
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  }, [result]);

  const exportCsv = () => {
    if (!result) return;
    const rows = [
      ["文件名", "类别编号", "类别名称", "置信度"],
      ...result.items.map((item) => [item.filename, item.class_id, item.class_name, item.confidence ?? ""]),
    ];
    const csv = rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(",")).join("\n");
    const blob = new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `traffic-sign-results-${Date.now()}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="page-stack">
      <div className="batch-layout">
        <div className="batch-main">
          <Card eyebrow="BATCH INPUT" title="选择图片文件" action={files.length > 0 && <Button variant="ghost" size="sm" icon={<Trash2 size={15} />} onClick={() => { setFiles([]); setResult(null); }}>清空列表</Button>}>
            <DropZone multiple files={files} onFiles={(next) => { setFiles(next.slice(0, 50)); setResult(null); mutation.reset(); }} />
            {files.length > 0 && (
              <div className="file-queue">
                <div className="file-queue__header"><span>待处理文件</span><strong>{files.length} / 50</strong></div>
                <div className="file-queue__items">
                  {files.slice(0, 8).map((file, index) => (
                    <div className="file-row" key={`${file.name}-${index}`}>
                      <span className="file-row__icon"><Files size={16} /></span>
                      <div><strong>{file.name}</strong><small>{(file.size / 1024).toFixed(1)} KB</small></div>
                      <button onClick={() => setFiles((items) => items.filter((_, itemIndex) => itemIndex !== index))} aria-label={`移除 ${file.name}`}><Trash2 size={15} /></button>
                    </div>
                  ))}
                  {files.length > 8 && <div className="queue-more">还有 {files.length - 8} 个文件未展开</div>}
                </div>
              </div>
            )}
            <div className="batch-actions">
              <span>批量接口会一次完成特征提取与模型推理，减少重复开销。</span>
              <Button icon={<Play size={17} />} loading={mutation.isPending} disabled={!files.length || !selectedModel} onClick={() => mutation.mutate()}>开始批量识别</Button>
            </div>
            {mutation.error && <div className="error-message">{mutation.error instanceof Error ? mutation.error.message : "批量识别失败"}</div>}
          </Card>

          <Card eyebrow="RESULT TABLE" title="处理结果" action={result && <Button variant="secondary" size="sm" icon={<Download size={15} />} onClick={exportCsv}>导出 CSV</Button>}>
            {result ? (
              <div className="table-wrap">
                <table className="data-table">
                  <thead><tr><th>文件名</th><th>识别类别</th><th>类别 ID</th><th>置信度</th><th>状态</th></tr></thead>
                  <tbody>{result.items.map((item, index) => (
                    <tr key={`${item.filename}-${index}`}>
                      <td><span className="table-file"><Files size={15} />{item.filename}</span></td>
                      <td><strong>{item.class_name}</strong></td>
                      <td>#{String(item.class_id).padStart(2, "0")}</td>
                      <td><span className="table-confidence"><i><b style={{ width: `${(item.confidence ?? 0) * 100}%` }} /></i>{formatPercent(item.confidence)}</span></td>
                      <td><span className="status-tag status-tag--success"><CheckCircle2 size={13} />完成</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            ) : (
              <div className="large-empty"><div><ListChecks size={34} /></div><strong>尚无批量处理结果</strong><p>选择多张交通标志图片，处理完成后可在这里查看并导出结果。</p></div>
            )}
          </Card>
        </div>

        <aside className="batch-aside">
          <Card eyebrow="BATCH SUMMARY" title="任务概览">
            <div className="summary-stack">
              <div><span className="summary-icon summary-icon--blue"><Files size={18} /></span><p>输入文件<strong>{files.length}</strong></p></div>
              <div><span className="summary-icon summary-icon--green"><CheckCircle2 size={18} /></span><p>成功处理<strong>{result?.count ?? 0}</strong></p></div>
              <div><span className="summary-icon summary-icon--red"><XCircle size={18} /></span><p>处理失败<strong>0</strong></p></div>
            </div>
            <div className="batch-timing"><span>总耗时</span><strong>{result ? formatDuration(result.predict_seconds) : "—"}</strong><small>{result ? `平均 ${formatDuration(result.predict_seconds / Math.max(result.count, 1))} / 张` : "任务完成后显示"}</small></div>
          </Card>
          <Card eyebrow="TOP CLASSES" title="主要类别">
            {summary.length ? <div className="class-summary-list">{summary.map(([name, count], index) => <div key={name}><span>{index + 1}</span><strong>{name}</strong><b>{count} 张</b></div>)}</div> : <div className="inline-empty">暂无类别统计</div>}
          </Card>
        </aside>
      </div>
    </div>
  );
}
