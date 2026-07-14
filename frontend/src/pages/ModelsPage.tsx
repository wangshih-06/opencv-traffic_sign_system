import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Box, CheckCircle2, Cpu, Database, HardDrive, Layers3, RefreshCw, ShieldCheck, Tag, Trash2 } from "lucide-react";
import clsx from "clsx";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { api } from "../lib/api";
import { formatBytes, formatPercent } from "../lib/format";
import { useAppStore } from "../store/useAppStore";

export function ModelsPage() {
  const queryClient = useQueryClient();
  const selectedModel = useAppStore((state) => state.selectedModel);
  const setSelectedModel = useAppStore((state) => state.setSelectedModel);
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.models });
  const labelsQuery = useQuery({ queryKey: ["labels"], queryFn: api.labels });

  const loadMutation = useMutation({
    mutationFn: (name: string) => api.loadModel(name),
    onSuccess: async (_, name) => {
      setSelectedModel(name);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["models"] }),
        queryClient.invalidateQueries({ queryKey: ["health"] }),
      ]);
    },
  });
  const cacheMutation = useMutation({
    mutationFn: (name: string) => api.clearCache(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const models = modelsQuery.data?.bundles ?? [];
  const active = models.find((model) => model.name === selectedModel) ?? models[0];

  return (
    <div className="page-stack">
      <div className="model-overview">
        <Card className="model-hero-card">
          <div className="model-hero-card__icon"><ShieldCheck size={30} /></div>
          <div><span className="eyebrow">ACTIVE INFERENCE ENGINE</span><h2>{active?.name.replace(".joblib", "") ?? "未发现模型"}</h2><p>经典机器学习分类器 · 标准化特征输入 · GTSRB 43 类交通标志</p></div>
          <span className={clsx("model-state", active && "model-state--ready")}><i />{active?.loaded ? "已加载" : active ? "可用" : "不可用"}</span>
        </Card>
        <div className="model-overview__metrics">
          <div><Cpu size={18} /><span>分类器</span><strong>{active?.classifier ?? "SVM"}</strong></div>
          <div><Layers3 size={18} /><span>特征模式</span><strong>{active?.feature_mode ?? "HOG + HSV"}</strong></div>
          <div><HardDrive size={18} /><span>模型体积</span><strong>{active ? formatBytes(active.size_bytes) : "—"}</strong></div>
          <div><Database size={18} /><span>缓存命中</span><strong>{active?.cache ? formatPercent(active.cache.hit_rate) : "—"}</strong></div>
        </div>
      </div>

      <div className="models-layout">
        <div className="models-main">
          <Card eyebrow="MODEL BUNDLES" title="可用模型" action={<Button variant="ghost" size="sm" icon={<RefreshCw size={15} />} onClick={() => modelsQuery.refetch()} loading={modelsQuery.isFetching}>刷新</Button>}>
            {models.length ? <div className="model-list">{models.map((model) => (
              <div className={clsx("model-row", selectedModel === model.name && "model-row--selected")} key={model.name}>
                <div className="model-row__icon"><Box size={22} /></div>
                <div className="model-row__main"><div><strong>{model.name.replace(".joblib", "")}</strong>{model.name === modelsQuery.data?.default_model && <span className="status-tag">默认</span>}{model.loaded && <span className="status-tag status-tag--success"><CheckCircle2 size={12} />已加载</span>}</div><p>{formatBytes(model.size_bytes)} · 更新于 {new Date(model.modified_at * 1000).toLocaleDateString("zh-CN")} · {model.feature_mode ?? "等待加载元数据"}</p></div>
                <div className="model-row__actions">
                  {model.loaded && <Button variant="ghost" size="sm" icon={<Trash2 size={14} />} loading={cacheMutation.isPending && cacheMutation.variables === model.name} onClick={() => cacheMutation.mutate(model.name)}>清缓存</Button>}
                  <Button variant={selectedModel === model.name ? "secondary" : "primary"} size="sm" loading={loadMutation.isPending && loadMutation.variables === model.name} onClick={() => loadMutation.mutate(model.name)}>{selectedModel === model.name && model.loaded ? "当前模型" : "加载模型"}</Button>
                </div>
              </div>
            ))}</div> : <div className="large-empty"><div><Box size={34} /></div><strong>模型目录为空</strong><p>请将训练后的 .joblib Bundle 放入 traffic_sign_system/models/artifacts。</p></div>}
            {(loadMutation.error || cacheMutation.error) && <div className="error-message">{(loadMutation.error ?? cacheMutation.error) instanceof Error ? (loadMutation.error ?? cacheMutation.error as Error)?.message : "模型操作失败"}</div>}
          </Card>

          <Card eyebrow="LABEL LIBRARY" title="交通标志类别库" action={<span className="count-badge">{labelsQuery.data?.length ?? 43} 类</span>}>
            <div className="label-grid">
              {(labelsQuery.data ?? []).map((label) => (
                <div className="label-item" key={label.class_id}><span>{String(label.class_id).padStart(2, "0")}</span><strong>{label.class_name}</strong></div>
              ))}
              {labelsQuery.isLoading && Array.from({ length: 12 }).map((_, index) => <div className="label-item label-item--loading" key={index} />)}
            </div>
          </Card>
        </div>

        <aside className="models-aside">
          <Card eyebrow="MODEL DETAIL" title="模型详情">
            <dl className="detail-list">
              <div><dt>Bundle 文件</dt><dd>{active?.name ?? "—"}</dd></div>
              <div><dt>分类器类型</dt><dd>{active?.classifier ?? "加载后获取"}</dd></div>
              <div><dt>特征维度</dt><dd>{active?.feature_dim?.toLocaleString() ?? "加载后获取"}</dd></div>
              <div><dt>特征模式</dt><dd>{active?.feature_mode ?? "HOG + HSV"}</dd></div>
              <div><dt>类别数量</dt><dd>{labelsQuery.data?.length ?? 43}</dd></div>
              <div><dt>缓存条目</dt><dd>{active?.cache ? `${active.cache.size} / ${active.cache.maxsize}` : "尚未初始化"}</dd></div>
            </dl>
          </Card>
          <Card eyebrow="PIPELINE" title="识别流程">
            <div className="pipeline-list">
              <div><span>01</span><p><strong>图像预处理</strong><small>尺寸归一化与灰度转换</small></p></div>
              <div><span>02</span><p><strong>特征融合</strong><small>HOG 形状 + HSV 颜色直方图</small></p></div>
              <div><span>03</span><p><strong>标准化</strong><small>使用训练期 Scaler 变换</small></p></div>
              <div><span>04</span><p><strong>分类输出</strong><small>SVM 概率与 Top-K 候选</small></p></div>
            </div>
          </Card>
          <Card eyebrow="SECURITY" title="本地安全">
            <div className="security-note"><Tag size={19} /><p><strong>数据不持久化上传</strong><span>图片仅在本机 FastAPI 进程内完成解码与推理，接口不会主动保存原图。</span></p></div>
          </Card>
        </aside>
      </div>
    </div>
  );
}
