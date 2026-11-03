import {
  Activity,
  ChartNoAxesCombined,
  FileText,
  FlaskConical,
  Menu,
  Orbit,
  X,
} from "lucide-react";

import { useState } from "react";
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
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        className="ct-menu-button"
        type="button"
        aria-label={isOpen ? "Close navigation menu" : "Open navigation menu"}
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        {isOpen ? <X size={21} /> : <Menu size={21} />}
      </button>
      {isOpen && <button className="ct-sidebar-overlay" type="button" aria-label="Close navigation menu" onClick={() => setIsOpen(false)} />}
    <aside className={`ct-sidebar${isOpen ? " is-open" : ""}`}>
      <NavLink className="ct-brand" to="/prediction" aria-label="CardioTox home" onClick={() => setIsOpen(false)}>
        <span className="ct-brand-mark"><Activity size={20} /></span>
        <span><strong>CardioTox</strong><small>MOLECULAR SAFETY</small></span>
      </NavLink>
      <div className="ct-nav-label">ANALYSIS</div>
      <nav className="ct-nav" aria-label="Analysis dashboards">
        {navigation.map(({ to, label, icon: Icon }, index) => (
          <NavLink key={to} to={to} onClick={() => setIsOpen(false)} className={({ isActive }) => `ct-nav-link${isActive ? " is-active" : ""}`}>
            <span className="ct-nav-index">0{index + 1}</span>
            <Icon size={17} strokeWidth={1.8} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
    </>
  );
}