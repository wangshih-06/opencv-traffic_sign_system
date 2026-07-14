import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  Boxes,
  ChevronDown,
  CircleHelp,
  Files,
  ImageIcon,
  Menu,
  Moon,
  Radio,
  Route,
  Settings,
  Sun,
  X,
} from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import clsx from "clsx";
import { api } from "../lib/api";
import { useAppStore } from "../store/useAppStore";

const navigation = [
  { to: "/", label: "单图识别", icon: ImageIcon, description: "分类与检测" },
  { to: "/realtime", label: "实时识别", icon: Radio, description: "视频与摄像头" },
  { to: "/batch", label: "批量处理", icon: Files, description: "多图片推理" },
  { to: "/analytics", label: "数据分析", icon: BarChart3, description: "趋势与统计" },
  { to: "/models", label: "模型管理", icon: Boxes, description: "Bundle 与类别" },
];

const pageMeta: Record<string, { title: string; subtitle: string }> = {
  "/": { title: "单图智能识别", subtitle: "上传图片，快速完成交通标志分类与候选区域检测" },
  "/realtime": { title: "实时流识别", subtitle: "连接摄像头或本地视频，实时返回识别结果" },
  "/batch": { title: "批量图片处理", subtitle: "一次上传多张图片并生成结构化识别结果" },
  "/analytics": { title: "识别数据分析", subtitle: "查看当前会话中的类别分布、置信度与效率趋势" },
  "/models": { title: "模型与类别管理", subtitle: "管理模型 Bundle、推理缓存和 43 类标志标签" },
};

export function AppShell() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const selectedModel = useAppStore((state) => state.selectedModel);
  const setSelectedModel = useAppStore((state) => state.setSelectedModel);
  const theme = useAppStore((state) => state.theme);
  const toggleTheme = useAppStore((state) => state.toggleTheme);

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.models });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const bundles = modelsQuery.data?.bundles;
    if (!selectedModel && bundles?.length) {
      const initial = modelsQuery.data?.active_model ?? modelsQuery.data?.default_model ?? bundles[0].name;
      setSelectedModel(initial);
    }
  }, [modelsQuery.data, selectedModel, setSelectedModel]);

  useEffect(() => setMobileOpen(false), [location.pathname]);

  const meta = pageMeta[location.pathname] ?? pageMeta["/"];
  const apiOnline = healthQuery.isSuccess && healthQuery.data.status === "ok";
  const modelOptions = useMemo(() => modelsQuery.data?.bundles ?? [], [modelsQuery.data]);

  return (
    <div className="app-shell">
      <aside className={clsx("sidebar", mobileOpen && "sidebar--open")}>
        <div className="sidebar__brand">
          <div className="brand-mark"><Route size={25} /></div>
          <div><strong>智路视界</strong><span>TRAFFIC VISION</span></div>
          <button className="icon-button sidebar__close" onClick={() => setMobileOpen(false)} aria-label="关闭菜单"><X size={19} /></button>
        </div>

        <div className="sidebar__label">识别工作台</div>
        <nav className="sidebar__nav">
          {navigation.map(({ to, label, icon: Icon, description }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => clsx("nav-item", isActive && "nav-item--active")}
            >
              <span className="nav-item__icon"><Icon size={19} /></span>
              <span className="nav-item__copy"><strong>{label}</strong><small>{description}</small></span>
              <span className="nav-item__dot" />
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="system-mini-card">
            <div className="system-mini-card__icon"><Settings size={18} /></div>
            <div><span>系统服务</span><strong>{apiOnline ? "运行正常" : healthQuery.isLoading ? "正在连接" : "连接失败"}</strong></div>
            <i className={clsx("status-dot", apiOnline && "status-dot--online")} />
          </div>
          <div className="sidebar__copyright">OpenCV · SVM · HOG + HSV</div>
        </div>
      </aside>
      {mobileOpen && <button className="sidebar-backdrop" onClick={() => setMobileOpen(false)} aria-label="关闭侧栏" />}

      <div className="app-main">
        <header className="topbar">
          <div className="topbar__title">
            <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="打开菜单"><Menu size={20} /></button>
            <div><h1>{meta.title}</h1><p>{meta.subtitle}</p></div>
          </div>
          <div className="topbar__actions">
            <div className="model-select-wrap">
              <span className="model-select-wrap__status"><i className={clsx("status-dot", selectedModel && "status-dot--online")} />当前模型</span>
              <select
                value={selectedModel ?? ""}
                onChange={(event) => setSelectedModel(event.target.value || null)}
                aria-label="选择识别模型"
              >
                {!modelOptions.length && <option value="">未发现模型</option>}
                {modelOptions.map((model) => <option key={model.name} value={model.name}>{model.name.replace(".joblib", "")}</option>)}
              </select>
              <ChevronDown size={15} />
            </div>
            <button className="icon-button" onClick={toggleTheme} aria-label="切换主题">
              {theme === "light" ? <Moon size={19} /> : <Sun size={19} />}
            </button>
            <button className="icon-button hide-mobile" aria-label="帮助"><CircleHelp size={19} /></button>
          </div>
        </header>

        {!apiOnline && !healthQuery.isLoading && (
          <div className="service-banner">
            <strong>后端服务未连接</strong>
            <span>请先运行 <code>python -m traffic_sign_system.api</code>，前端会自动重试。</span>
          </div>
        )}

        <main className="page-content"><Outlet /></main>
        <footer className="statusbar">
          <span><i className={clsx("status-dot", apiOnline && "status-dot--online")} />{apiOnline ? "API 服务在线" : "API 服务离线"}</span>
          <span>模型：{selectedModel?.replace(".joblib", "") ?? "未选择"}</span>
          <span>类别：{healthQuery.data?.labels ?? 43}</span>
          <span className="statusbar__version">Web UI v1.0</span>
        </footer>
      </div>
    </div>
  );
}
