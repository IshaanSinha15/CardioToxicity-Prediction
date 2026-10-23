import {
  Activity,
  ChartNoAxesCombined,
  FileText,
  FlaskConical,
  Orbit,
} from "lucide-react";

import {
  NavLink,
} from "react-router-dom";


const navigation = [
  {
    to: "/prediction",
    label: "Regression",
    icon: FlaskConical,
  },
  {
    to: "/ord",
    label: "ORD simulation",
    icon: Orbit,
  },
  {
    to: "/xai",
    label: "XAI & similarity",
    icon: ChartNoAxesCombined,
  },
  {
    to: "/report",
    label: "Combined report",
    icon: FileText,
  },
];


export default function Sidebar() {

  return (
    <aside className="ct-sidebar">
      <NavLink className="ct-brand" to="/prediction" aria-label="CardioTox home">
        <span className="ct-brand-mark"><Activity size={20} /></span>
        <span><strong>CardioTox</strong><small>MOLECULAR SAFETY</small></span>
      </NavLink>
      <div className="ct-nav-label">ANALYSIS</div>
      <nav className="ct-nav" aria-label="Analysis dashboards">
        {navigation.map(({ to, label, icon: Icon }, index) => (
          <NavLink key={to} to={to} className={({ isActive }) => `ct-nav-link${isActive ? " is-active" : ""}`}>
            <span className="ct-nav-index">0{index + 1}</span>
            <Icon size={17} strokeWidth={1.8} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="ct-sidebar-bottom">
        <span className="ct-live-dot" />
        <span><strong>API connection</strong><small>Local pipeline service</small></span>
      </div>
    </aside>
  );
}