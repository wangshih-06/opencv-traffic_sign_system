import { useMemo } from "react";
import { Activity, BarChart3, Clock3, Gauge, PieChart as PieIcon, Trash2, TrendingUp } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { formatPercent, formatTime } from "../lib/format";
import { useAppStore } from "../store/useAppStore";

const COLORS = ["#3b82f6", "#22b8cf", "#818cf8", "#38bdf8", "#34d399", "#fbbf24"];

export function AnalyticsPage() {
  const history = useAppStore((state) => state.history);
  const clearHistory = useAppStore((state) => state.clearHistory);

  const analytics = useMemo(() => {
    const validConfidence = history.filter((item) => item.confidence != null);
    const averageConfidence = validConfidence.length
      ? validConfidence.reduce((sum, item) => sum + (item.confidence ?? 0), 0) / validConfidence.length
      : 0;
    const averageDuration = history.length
      ? history.reduce((sum, item) => sum + item.duration_ms, 0) / history.length
      : 0;
    const highConfidence = validConfidence.filter((item) => (item.confidence ?? 0) >= 0.85).length;
    const categoryCounts = new Map<string, number>();
    history.forEach((item) => categoryCounts.set(item.class_name, (categoryCounts.get(item.class_name) ?? 0) + 1));
    const categories = [...categoryCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, value]) => ({ name, value }));
    const trend = [...history].reverse().slice(-24).map((item, index) => ({
      index: index + 1,
      time: formatTime(item.timestamp),
      confidence: Math.round((item.confidence ?? 0) * 100),
      duration: Math.round(item.duration_ms),
      name: item.class_name,
    }));
    const sources = ["image", "camera", "video", "batch"].map((source) => ({
      source: { image: "单图", camera: "摄像头", video: "视频", batch: "批量" }[source]!,
      count: history.filter((item) => item.source === source).length,
    }));
    return { averageConfidence, averageDuration, highConfidence, categories, trend, sources };
  }, [history]);

  return (
    <div className="page-stack">
      <div className="analytics-hero">
        <div><span className="eyebrow">SESSION INSIGHTS</span><h2>当前会话识别概览</h2><p>数据仅保存在浏览器本地，帮助快速评估模型在当前样本上的识别表现。</p></div>
        {history.length > 0 && <Button variant="secondary" icon={<Trash2 size={16} />} onClick={clearHistory}>清空会话数据</Button>}
      </div>

      <div className="analytics-kpis">
        <div className="analytics-kpi"><span className="analytics-kpi__icon blue"><Activity size={20} /></span><div><small>识别总次数</small><strong>{history.length}</strong><p>本地会话累计</p></div></div>
        <div className="analytics-kpi"><span className="analytics-kpi__icon cyan"><Gauge size={20} /></span><div><small>平均置信度</small><strong>{formatPercent(analytics.averageConfidence)}</strong><p>有效概率结果</p></div></div>
        <div className="analytics-kpi"><span className="analytics-kpi__icon violet"><Clock3 size={20} /></span><div><small>平均推理耗时</small><strong>{analytics.averageDuration ? `${analytics.averageDuration.toFixed(0)} ms` : "—"}</strong><p>包含单图与批量</p></div></div>
        <div className="analytics-kpi"><span className="analytics-kpi__icon green"><TrendingUp size={20} /></span><div><small>高置信结果</small><strong>{analytics.highConfidence}</strong><p>置信度 ≥ 85%</p></div></div>
      </div>

      {history.length ? (
        <div className="analytics-grid">
          <Card className="chart-card chart-card--wide" eyebrow="CONFIDENCE TREND" title="置信度与推理耗时趋势">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={analytics.trend} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
                  <defs><linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/><stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/></linearGradient></defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="var(--chart-grid)" vertical={false} />
                  <XAxis dataKey="index" tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)", boxShadow: "var(--shadow)" }} formatter={(value) => [`${value}%`, "置信度"]} labelFormatter={(_, payload) => payload?.[0]?.payload?.name ?? ""} />
                  <Area type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2.5} fill="url(#confidenceFill)" activeDot={{ r: 5 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="chart-card" eyebrow="CLASS DISTRIBUTION" title="类别分布">
            <div className="donut-layout">
              <div className="donut-chart">
                <ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={analytics.categories} dataKey="value" nameKey="name" innerRadius={54} outerRadius={78} paddingAngle={3}>{analytics.categories.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip contentStyle={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)" }} /></PieChart></ResponsiveContainer>
                <div className="donut-center"><strong>{analytics.categories.length}</strong><span>识别类别</span></div>
              </div>
              <div className="chart-legend">{analytics.categories.slice(0, 5).map((item, index) => <div key={item.name}><i style={{ background: COLORS[index % COLORS.length] }} /><span>{item.name}</span><strong>{item.value}</strong></div>)}</div>
            </div>
          </Card>

          <Card className="chart-card" eyebrow="SOURCE MIX" title="识别来源">
            <div className="chart-wrap chart-wrap--short"><ResponsiveContainer width="100%" height="100%"><BarChart data={analytics.sources} margin={{ top: 8, right: 0, left: -28, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" stroke="var(--chart-grid)" vertical={false}/><XAxis dataKey="source" axisLine={false} tickLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }}/><YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }}/><Tooltip cursor={{ fill: "var(--soft-blue)" }} contentStyle={{ border: "1px solid var(--border)", borderRadius: 12, background: "var(--panel)" }}/><Bar dataKey="count" name="识别次数" fill="#38bdf8" radius={[7, 7, 2, 2]} maxBarSize={44}/></BarChart></ResponsiveContainer></div>
          </Card>

          <Card className="chart-card chart-card--wide" eyebrow="RECENT RECORDS" title="最近记录">
            <div className="table-wrap"><table className="data-table"><thead><tr><th>时间</th><th>来源</th><th>文件</th><th>识别类别</th><th>置信度</th><th>耗时</th></tr></thead><tbody>{history.slice(0, 10).map((item) => <tr key={item.id}><td>{formatTime(item.timestamp)}</td><td><span className="source-tag">{{ image: "单图", camera: "摄像头", video: "视频", batch: "批量" }[item.source]}</span></td><td>{item.filename}</td><td><strong>{item.class_name}</strong></td><td>{formatPercent(item.confidence)}</td><td>{item.duration_ms.toFixed(0)} ms</td></tr>)}</tbody></table></div>
          </Card>
        </div>
      ) : (
        <Card><div className="analytics-empty"><div className="analytics-empty__visual"><BarChart3 size={42} /><span /><span /></div><h3>还没有可分析的识别数据</h3><p>前往单图识别、实时识别或批量处理页面完成一次推理后，这里会自动生成趋势与统计图表。</p><div className="analytics-empty__features"><span><Gauge size={17} />置信度趋势</span><span><PieIcon size={17} />类别分布</span><span><Clock3 size={17} />性能统计</span></div></div></Card>
      )}
    </div>
  );
}
