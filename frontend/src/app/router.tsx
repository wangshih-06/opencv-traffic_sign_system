import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "../components/AppShell";

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", lazy: async () => ({ Component: (await import("../pages/ImagePage")).ImagePage }) },
      { path: "/realtime", lazy: async () => ({ Component: (await import("../pages/RealtimePage")).RealtimePage }) },
      { path: "/batch", lazy: async () => ({ Component: (await import("../pages/BatchPage")).BatchPage }) },
      { path: "/feedback", lazy: async () => ({ Component: (await import("../pages/FeedbackPage")).FeedbackPage }) },
      { path: "/analytics", lazy: async () => ({ Component: (await import("../pages/AnalyticsPage")).AnalyticsPage }) },
      { path: "/models", lazy: async () => ({ Component: (await import("../pages/ModelsPage")).ModelsPage }) },
    ],
  },
]);
