import { ArrowLeft, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { usePipelineResult } from "../context/usePipelineResult";

function artifactUrl(path?: string) {
  return path ? `${import.meta.env.VITE_API_BASE_URL ?? ""}${path}` : undefined;
}

export default function ORDDashboard() {
  const { result } = usePipelineResult();
  const simulation = result?.simulation;
  const features = simulation?.features ?? {};
  const plot = artifactUrl(result?.artifact_urls?.ord?.voltage_plot);
  const simulationError = simulation?.error;
  const simulationMessage = simulationError?.includes("Microsoft Visual C++")
    ? "ORd needs Microsoft C++ Build Tools 14 or newer to compile Myokit's simulation on Windows. Install the compiler, then rerun the analysis."
    : simulationError?.split("\n")[0] ?? "The ORd voltage plot was not produced. Check the Myokit/CellML setup, then rerun the analysis.";
  const metrics = [
    ["Resting potential", features.RMP, "mV"],
    ["Peak voltage", features.Peak, "mV"],
    ["APD50", features.APD50, "ms"],
    ["APD90", features.APD90, "ms"],
    ["Triangulation", features.Triangulation, "ms"],
  ] as const;

  return (
    <div className="ct-page">
      <div className="ct-page-heading">
        <div><p className="ct-eyebrow">02 / ELECTROPHYSIOLOGY</p><h1>ORd simulation</h1><p className="ct-lede">Translate predicted ion-channel block into a cellular action-potential response.</p></div>
        <span className="ct-heading-mark">02<span>—</span>04</span>
      </div>
      {!result ? (
        <section className="ct-panel ct-empty-dashboard"><h2>Simulation needs a pipeline result</h2><p>Start with a SMILES string and concentration to run regression and the ORd model.</p><Link className="ct-float-button" to="/prediction">Go to regression <ArrowRight size={16} /></Link></section>
      ) : (
        <>
          <section className="ct-panel ct-chart-panel">
            <div className="ct-panel-heading"><div><p className="ct-eyebrow">ACTION POTENTIAL</p><h2>Membrane voltage over time</h2></div><span className={`ct-status ${simulation?.status === "complete" ? "is-complete" : ""}`}>{simulation?.status ?? "not available"}</span></div>
            {plot ? <img className="ct-result-image" src={plot} alt="ORd simulated membrane voltage waveform" /> : <div className="ct-empty-state"><p>{simulationMessage}</p></div>}
          </section>
          <section className="ct-panel ct-metrics-panel">
            <div className="ct-panel-heading"><div><p className="ct-eyebrow">EXTRACTED FEATURES</p><h2>Action-potential measurements</h2></div><span className="ct-molecule-label">{result.input.smiles}</span></div>
            <div className="ct-metric-grid">{metrics.map(([label, value, unit]) => <article className="ct-metric" key={label}><span>{label}</span><strong>{typeof value === "number" ? value.toFixed(2) : "--"}</strong><small>{unit}</small></article>)}</div>
          </section>
          <div className="ct-page-actions"><Link className="ct-float-button ct-button-light" to="/prediction"><ArrowLeft size={16} /> Back to regression</Link><Link className="ct-float-button" to="/xai">Review explanations <ArrowRight size={16} /></Link></div>
        </>
      )}
    </div>
  );
}