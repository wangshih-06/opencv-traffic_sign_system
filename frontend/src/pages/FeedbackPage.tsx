import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, CircleAlert, Download, Flag, ListTodo, MessageSquareWarning, Send, SlidersHorizontal, Trash2 } from "lucide-react";
import clsx from "clsx";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { api } from "../lib/api";
import { formatPercent, formatTime } from "../lib/format";
import type { FeedbackStatus, FeedbackVerdict, HistoryItem } from "../lib/types";
import { useAppStore } from "../store/useAppStore";

type ReviewTarget = {
  key: string;
  historyId: string;
  queueId?: string;
  source: HistoryItem["source"];
  filename: string;
  model?: string | null;
  class_id: number;
  class_name: string;
  confidence: number | null;
  timestamp: number;
  automatic: boolean;
};

const statusLabels: Record<FeedbackStatus, string> = {
  new: "待复核",
  reviewed: "已复核",
  exported: "已导出",
};

const sourceLabels: Record<HistoryItem["source"], string> = {
  image: "单图",
  camera: "摄像头",
  video: "视频",
  batch: "批量",
};

export function FeedbackPage() {
  const queryClient = useQueryClient();
  const history = useAppStore((state) => state.history);
  const selectedModel = useAppStore((state) => state.selectedModel);
  const reviewConfidenceThreshold = useAppStore((state) => state.reviewConfidenceThreshold);
  const setReviewConfidenceThreshold = useAppStore((state) => state.setReviewConfidenceThreshold);
  const [selectedTargetKey, setSelectedTargetKey] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<FeedbackVerdict>("incorrect");
  const [correctedClassId, setCorrectedClassId] = useState("");
  const [note, setNote] = useState("");
  const [feedbackFilter, setFeedbackFilter] = useState<FeedbackStatus | "all">("all");

  const labelsQuery = useQuery({ queryKey: ["labels"], queryFn: api.labels });
  const feedbackQuery = useQuery({
    queryKey: ["feedback"],
    queryFn: () => api.feedback("all"),
  });
  const reviewQueueQuery = useQuery({
    queryKey: ["review-queue", "pending"],
    queryFn: () => api.reviewQueue("pending"),
  });
  const createMutation = useMutation({
    mutationFn: () => {
      if (!selectedTarget) throw new Error("请先选择一条待复核记录");
      const correctedId = verdict === "correct" ? selectedTarget.class_id : Number(correctedClassId);
      if (verdict === "incorrect" && !Number.isInteger(correctedId)) {
        throw new Error("请选择纠正后的真实类别");
      }
      return api.createFeedback({
        history_id: selectedTarget.historyId,
        source: selectedTarget.source,
        filename: selectedTarget.filename,
        model: selectedTarget.model ?? selectedModel,
        predicted_class_id: selectedTarget.class_id,
        predicted_class_name: selectedTarget.class_name,
        predicted_confidence: selectedTarget.confidence,
        corrected_class_id: correctedId,
        verdict,
        note,
        review_queue_id: selectedTarget.queueId,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feedback"] });
      queryClient.invalidateQueries({ queryKey: ["review-queue"] });
      setNote("");
      setVerdict("incorrect");
    },
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: FeedbackStatus }) => api.updateFeedbackStatus(id, status),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["feedback"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteFeedback(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["feedback"] }),
  });

  const allFeedback = feedbackQuery.data?.items ?? [];
  const visibleFeedback = feedbackFilter === "all"
    ? allFeedback
    : allFeedback.filter((item) => item.status === feedbackFilter);
  const feedbackIds = useMemo(
    () => new Set(allFeedback.map((item) => item.history_id).filter(Boolean)),
    [allFeedback],
  );
  const automaticTargets = useMemo<ReviewTarget[]>(() => (reviewQueueQuery.data?.items ?? []).map((item) => ({
    key: `queue:${item.id}`,
    historyId: item.history_id,
    queueId: item.id,
    source: item.source,
    filename: item.filename,
    model: item.model,
    class_id: item.predicted_class_id,
    class_name: item.predicted_class_name,
    confidence: item.predicted_confidence,
    timestamp: new Date(item.created_at).getTime(),
    automatic: true,
  })), [reviewQueueQuery.data]);
  const queuedHistoryIds = useMemo(() => new Set(automaticTargets.map((item) => item.historyId)), [automaticTargets]);
  const manualTargets = useMemo<ReviewTarget[]>(() => history
    .filter((item) => !feedbackIds.has(item.id) && !queuedHistoryIds.has(item.id))
    .map((item) => ({ ...item, key: `history:${item.id}`, historyId: item.id, automatic: false })), [feedbackIds, history, queuedHistoryIds]);
  const reviewTargets = useMemo(
    () => [...automaticTargets, ...manualTargets].sort((a, b) => b.timestamp - a.timestamp),
    [automaticTargets, manualTargets],
  );
  const selectedTarget = reviewTargets.find((item) => item.key === selectedTargetKey) ?? reviewTargets[0] ?? null;

  useEffect(() => {
    if (!selectedTargetKey || !reviewTargets.some((item) => item.key === selectedTargetKey)) {
      setSelectedTargetKey(reviewTargets[0]?.key ?? null);
    }
  }, [reviewTargets, selectedTargetKey]);

  useEffect(() => {
    setCorrectedClassId(selectedTarget ? String(selectedTarget.class_id) : "");
  }, [selectedTarget?.key, selectedTarget?.class_id]);

  const mutationError = [createMutation.error, statusMutation.error, deleteMutation.error]
    .find((error): error is Error => error instanceof Error)?.message;

  const handleExport = () => {
    window.location.href = "/api/feedback/export?status=reviewed";
  };

  return (
    <div className="page-stack feedback-page">
      <div className="feedback-hero">
        <div>
          <span className="eyebrow">HUMAN-IN-THE-LOOP</span>
          <h2>错例纠正与反馈闭环</h2>
          <p>低于设定阈值的识别结果会自动进入队列；人工确认后可导出为训练纠正样本。</p>
        </div>
        <div className="feedback-hero__actions">
          <label className="review-threshold"><SlidersHorizontal size={15} /><span>自动入队阈值</span><select value={reviewConfidenceThreshold} onChange={(event) => setReviewConfidenceThreshold(Number(event.target.value))}><option value={0.6}>60%</option><option value={0.7}>70%</option><option value={0.8}>80%</option><option value={0.9}>90%</option></select></label>
          <Button variant="secondary" icon={<Download size={16} />} onClick={handleExport}>导出已复核样本</Button>
        </div>
      </div>

      <div className="feedback-kpis">
        <div className="feedback-kpi"><span><MessageSquareWarning size={19} /></span><div><small>反馈总数</small><strong>{feedbackQuery.data?.stats.total ?? 0}</strong></div></div>
        <div className="feedback-kpi feedback-kpi--danger"><span><CircleAlert size={19} /></span><div><small>错例数量</small><strong>{feedbackQuery.data?.stats.incorrect ?? 0}</strong></div></div>
        <div className="feedback-kpi feedback-kpi--warning"><span><ListTodo size={19} /></span><div><small>自动待复核</small><strong>{reviewQueueQuery.data?.stats.pending ?? 0}</strong></div></div>
        <div className="feedback-kpi feedback-kpi--success"><span><CheckCircle2 size={19} /></span><div><small>已形成闭环</small><strong>{feedbackQuery.data?.stats.exported ?? 0}</strong></div></div>
      </div>

      <div className="feedback-grid">
        <div className="feedback-main">
          <Card eyebrow="CORRECTION QUEUE" title="待人工复核（低置信度优先）">
            {reviewTargets.length ? (
              <div className="feedback-history-list">
                {reviewTargets.slice(0, 30).map((item) => (
                  <button
                    key={item.key}
                    className={clsx("feedback-history-item", selectedTarget?.key === item.key && "feedback-history-item--selected", item.automatic && "feedback-history-item--automatic")}
                    onClick={() => {
                      setSelectedTargetKey(item.key);
                      setCorrectedClassId(String(item.class_id));
                    }}
                  >
                    <span className="feedback-history-item__icon">{item.automatic ? <ListTodo size={15} /> : <Flag size={15} />}</span>
                    <span className="feedback-history-item__copy"><strong>{item.class_name}</strong><small>{item.filename} · {sourceLabels[item.source]} · {formatTime(item.timestamp)}{item.automatic ? " · 自动入队" : ""}</small></span>
                    <b>{formatPercent(item.confidence)}</b>
                  </button>
                ))}
              </div>
            ) : (
              <div className="feedback-empty"><MessageSquareWarning size={32} /><strong>暂无待反馈识别记录</strong><p>先前往单图、实时或批量识别页面完成一次推理，再回到这里进行人工纠正。</p></div>
            )}
          </Card>

          <Card eyebrow="HUMAN REVIEW" title="提交人工判断">
            {selectedTarget ? (
              <div className="feedback-form">
                <div className="feedback-original">
                  <span>模型原始判断</span>
                  <strong>{selectedTarget.class_name}</strong>
                  <small>类别 #{selectedTarget.class_id} · 置信度 {formatPercent(selectedTarget.confidence)}{selectedTarget.automatic ? " · 已由低置信度规则自动入队" : ""}</small>
                </div>
                <div className="feedback-verdict" role="radiogroup" aria-label="判断识别是否正确">
                  <button className={clsx(verdict === "correct" && "active", "feedback-verdict--correct")} onClick={() => setVerdict("correct")}><CheckCircle2 size={17} />识别正确</button>
                  <button className={clsx(verdict === "incorrect" && "active", "feedback-verdict--incorrect")} onClick={() => setVerdict("incorrect")}><CircleAlert size={17} />标记错例</button>
                </div>
                {verdict === "incorrect" && (
                  <label className="feedback-field"><span>真实类别</span><select value={correctedClassId} onChange={(event) => setCorrectedClassId(event.target.value)}><option value="">请选择人工确认类别</option>{(labelsQuery.data ?? []).map((label) => <option key={label.class_id} value={label.class_id}>{label.class_id} · {label.class_name}</option>)}</select></label>
                )}
                <label className="feedback-field"><span>备注（可选）</span><textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={1000} placeholder="例如：遮挡、反光、类别相似或候选框偏移" /></label>
                {mutationError && <div className="error-message">{mutationError}</div>}
                {createMutation.isSuccess && <div className="feedback-success"><CheckCircle2 size={16} />反馈已保存，可在下方复核并导出。</div>}
                <Button icon={<Send size={16} />} loading={createMutation.isPending} onClick={() => createMutation.mutate()} disabled={verdict === "incorrect" && !correctedClassId}>提交反馈</Button>
              </div>
            ) : <div className="feedback-empty"><Flag size={32} /><strong>请选择识别记录</strong><p>左侧列表中的识别结果会自动带入纠正表单。</p></div>}
          </Card>
        </div>

        <aside className="feedback-side">
          <Card eyebrow="FEEDBACK PIPELINE" title="处理流程">
            <div className="feedback-steps"><div className="feedback-step feedback-step--active"><b>1</b><span><strong>采集</strong><small>标记正确或错例</small></span></div><div className="feedback-step"><b>2</b><span><strong>复核</strong><small>检查人工纠正标签</small></span></div><div className="feedback-step"><b>3</b><span><strong>导出</strong><small>生成训练数据 CSV</small></span></div></div>
          </Card>
          <Card eyebrow="DATA QUALITY" title="质量建议"><ul className="tips-list"><li>优先反馈低置信度和类别混淆样本</li><li>纠正类别必须与真实标志一致</li><li>备注遮挡、反光等原因便于后续分析</li><li>复核后导出 CSV，再加入训练集重训</li></ul></Card>
        </aside>
      </div>

      <Card eyebrow="REVIEW HISTORY" title="已提交反馈" action={<select className="feedback-filter" value={feedbackFilter} onChange={(event) => setFeedbackFilter(event.target.value as FeedbackStatus | "all")}><option value="all">全部状态</option><option value="new">待复核</option><option value="reviewed">已复核</option><option value="exported">已导出</option></select>}>
        {visibleFeedback.length ? (
          <div className="feedback-table-wrap"><table className="data-table feedback-table"><thead><tr><th>时间</th><th>原始识别</th><th>人工标签</th><th>判断</th><th>状态</th><th>操作</th></tr></thead><tbody>{visibleFeedback.map((item) => <tr key={item.id}><td>{new Date(item.created_at).toLocaleString("zh-CN", { hour: "2-digit", minute: "2-digit" })}</td><td><strong>{item.predicted_class_name}</strong><small>{item.filename}</small></td><td><strong className={item.verdict === "incorrect" ? "feedback-corrected" : ""}>{item.corrected_class_name}</strong><small>{item.note || "—"}</small></td><td><span className={clsx("feedback-verdict-tag", `feedback-verdict-tag--${item.verdict}`)}>{item.verdict === "incorrect" ? "错例" : "正确"}</span></td><td><span className={clsx("feedback-status-tag", `feedback-status-tag--${item.status}`)}>{statusLabels[item.status]}</span></td><td><div className="feedback-row-actions">{item.status === "new" && <Button size="sm" variant="secondary" onClick={() => statusMutation.mutate({ id: item.id, status: "reviewed" })}>标记已复核</Button>}{item.status === "reviewed" && <Button size="sm" variant="secondary" onClick={() => statusMutation.mutate({ id: item.id, status: "exported" })}>标记已导出</Button>}<button className="icon-button" aria-label="删除反馈" onClick={() => deleteMutation.mutate(item.id)}><Trash2 size={15} /></button></div></td></tr>)}</tbody></table></div>
        ) : <div className="feedback-empty feedback-empty--row"><MessageSquareWarning size={29} /><strong>还没有提交过反馈</strong><p>提交后的人工标签会在这里形成闭环追踪。</p></div>}
      </Card>
    </div>
  );
}
