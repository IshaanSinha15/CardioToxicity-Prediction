import { Outlet } from "react-router-dom";

import Sidebar from "../components/common/Sidebar";
import useApi from "../../hooks/useApi";

export default function DashboardLayout() {
  const { loading, online } = useApi();
  return (
    <div className="ct-shell">
      <Sidebar />
      <div className="ct-main">
        <header className="ct-topbar">
          <span>Cardiac safety workspace</span>
          <span className={`ct-topbar-status${online ? " is-online" : " is-offline"}`}><i /> {loading ? "Checking API" : online ? "API connected" : "API offline"}</span>
        </header>
        <main className="ct-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}