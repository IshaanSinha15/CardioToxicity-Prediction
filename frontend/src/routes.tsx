// src/routes.tsx

import { createBrowserRouter } from "react-router-dom";

import PredictionPage from "./pages/PredictionPage";
import HistoryPage from "./pages/HistoryPage";
import DocumentationPage from "./pages/DocumentationPage";
import AboutPage from "./pages/AboutPage";
import NotFound from "./pages/NotFound";

const router = createBrowserRouter([
  {
    path: "/",
    element: <PredictionPage />,
  },
  {
    path: "/prediction",
    element: <PredictionPage />,
  },
  {
    path: "/history",
    element: <HistoryPage />,
  },
  {
    path: "/documentation",
    element: <DocumentationPage />,
  },
  {
    path: "/about",
    element: <AboutPage />,
  },
  {
    path: "*",
    element: <NotFound />,
  },
]);

export default router;