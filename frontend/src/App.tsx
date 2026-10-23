import "./App.css";

import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import DashboardLayout from "./layouts/DashboardLayout";

import PredictionPage from "./pages/RegressionDashboard";
import ORDDashboard from "./pages/ORDDashboard";
import XAIDashboard from "./pages/XAIDashboard";
import CombinedReport from "./pages/CombinedReport";
import { PredictionProvider } from "./context/PredictionContext";

// Keep these imports if these files already exist
import HistoryPage from "./pages/HistoryPage";
import DocumentationPage from "./pages/DocumentationPage";
import AboutPage from "./pages/AboutPage";


export default function App() {
  return (
    <PredictionProvider>
    <BrowserRouter>

      <Routes>
        <Route element={<DashboardLayout />}>

          {/* =================================================
              DEFAULT ROUTE
             ================================================= */}

          <Route
            index
            element={
              <Navigate
                to="/prediction"
                replace
              />
            }
          />


          {/* =================================================
              PREDICTION
             ================================================= */}

          <Route
            path="/prediction"
            element={
              <PredictionPage />
            }
          />

          <Route path="/ord" element={<ORDDashboard />} />
          <Route path="/xai" element={<XAIDashboard />} />
          <Route path="/report" element={<CombinedReport />} />


          {/* =================================================
              HISTORY
             ================================================= */}

          <Route
            path="/history"
            element={
              <HistoryPage />
            }
          />


          {/* =================================================
              DOCUMENTATION
             ================================================= */}

          <Route
            path="/documentation"
            element={
              <DocumentationPage />
            }
          />


          {/* =================================================
              ABOUT
             ================================================= */}

          <Route
            path="/about"
            element={
              <AboutPage />
            }
          />


          {/* =================================================
              FALLBACK
             ================================================= */}

          <Route
            path="*"
            element={
              <Navigate
                to="/prediction"
                replace
              />
            }
          />

        </Route>
      </Routes>

    </BrowserRouter>
    </PredictionProvider>
  );
}