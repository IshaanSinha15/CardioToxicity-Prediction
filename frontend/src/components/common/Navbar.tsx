import { Activity } from "lucide-react";
import {
  Link,
  useLocation,
} from "react-router-dom";


const pageNames: Record<string, string> = {
  "/": "Prediction",
  "/prediction": "Prediction",
  "/history": "History",
  "/documentation": "Documentation",
  "/about": "About",
};


export default function Navbar() {

  const location =
    useLocation();


  const pageName =
    pageNames[
      location.pathname
    ] ?? "CardioTox";


  return (
    <header className="cardiotox-navbar">

      <div className="navbar-page">

        <Activity
          size={18}
          strokeWidth={2}
        />

        <span>
          {pageName}
        </span>

      </div>


      <div className="navbar-actions">

        <div className="navbar-status">

          <span className="status-dot" />

          <span>
            CardioTox-v1 · Online
          </span>

        </div>


        <Link
          to="/documentation"
          className="navbar-link"
        >
          Docs
        </Link>


        <Link
          to="/about"
          className="navbar-link"
        >
          About
        </Link>

      </div>

    </header>
  );
}