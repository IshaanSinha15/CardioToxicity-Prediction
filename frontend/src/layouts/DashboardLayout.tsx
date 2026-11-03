import { Outlet } from "react-router-dom";

import Sidebar from "../components/common/Sidebar";

export default function DashboardLayout() {
  return (
    <div className="ct-shell">
      <Sidebar />
      <div className="ct-main">
        <main className="ct-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}