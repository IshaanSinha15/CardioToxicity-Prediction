import { ArrowLeft, Download, RotateCcw } from "lucide-react";
import { Link } from "react-router-dom";
import { usePipelineResult } from "../context/usePipelineResult";

const channelRows = [
  ["hERG / IKr", "herg", "herg_block", "hERG"],
  ["NaV1.5 / INa", "nav", "nav_block", "Nav1.5"],
  ["CaV1.2 / ICaL", "cav", "cav_block", "Cav1.2"],
] as const;

function artifactUrl(result: NonNullable<ReturnType<typeof usePipelineResult>["result"]>, group: string, name: string) {
  const path = result.artifact_urls?.[group]?.[name];
  return path ? `${import.meta.env.VITE_API_BASE_URL ?? ""}${path}` : undefined;
}

export default function CombinedReport() {
  const { result, setResult } = usePipelineResult();
  const interpretation = result?.frontend?.interpretation as { summary?: string; details?: string[] } | undefined;

  function download() {
    if (!result) return;
    const file = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "cardiotox-analysis-report.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="ct-page">
      <div className="ct-page-heading"><div><p className="ct-eyebrow">04 / REPORT</p><h1>Combined analysis</h1><p className="ct-lede">A single review surface for regression, cellular simulation, classification and reference evidence.</p></div><span className="ct-heading-mark">04<span>—</span>04</span></div>
      {!result ? (
        <section className="ct-panel ct-empty-dashboard"><h2>Report not generated</h2><p>Complete a pipeline run to assemble the combined molecule report.</p><Link className="ct-float-button" to="/prediction">Start analysis <span>↗</span></Link></section>
      ) : (
        <>
          <section className="ct-report-intro">
            <div><p className="ct-eyebrow">SUBMITTED MOLECULE</p><h2>{result.input.drug_name || "Untitled molecule"}</h2><code>{result.input.smiles}</code></div>
            <div className="ct-report-dose"><span>Concentration</span><strong>{result.input.concentration_nm ?? result.input.dose_nm ?? "--"}<small> nM</small></strong></div>
          </section>

          <section className="ct-panel ct-report-table-panel">
            <div className="ct-panel-heading"><div><p className="ct-eyebrow">REGRESSION & BLOCK</p><h2>Channel summary</h2></div><span className="ct-status is-complete">{result.mechanistic_risk?.level ? String(result.mechanistic_risk.level) : result.simulation?.status ?? "analysis complete"}</span></div>
            <div className="ct-table-wrap"><table className="ct-table"><thead><tr><th>Channel</th><th>Predicted IC50</th><th>Block at dose</th><th>Risk band</th></tr></thead><tbody>{channelRows.map(([label, key, block, riskKey]) => {
              const ic50 = result.ic50_prediction?.[key]?.IC50_nM;
              const percentage = result.dose_response?.[block];
              const band = result.mechanistic_classification?.channels?.[riskKey]?.band;
              return <tr key={key}><td>{label}</td><td>{typeof ic50 === "number" ? `${ic50.toLocaleString(undefined, { maximumFractionDigits: 2 })} nM` : "--"}</td><td>{typeof percentage === "number" ? `${(percentage <= 1 ? percentage * 100 : percentage).toFixed(1)}%` : "--"}</td><td>{band ?? "Unavailable"}</td></tr>;
            })}</tbody></table></div>
          </section>

          <section className="ct-report-lower-grid">
            <article className="ct-panel"><p className="ct-eyebrow">INTERPRETATION</p><h2>Evidence summary</h2><p className="ct-report-summary">{interpretation?.summary ?? "The available evidence is summarized across channel potency, block, simulation, classification and similarity."}</p>{interpretation?.details?.map((detail) => <p className="ct-report-detail" key={detail}>{detail}</p>)}<p className="ct-report-summary">ORD simulation: <strong>{result.simulation?.status ?? "not reported"}</strong>. Classification: <strong>{result.classification?.predicted_class ?? "not reported"}</strong>.</p></article>
            <article className="ct-panel"><p className="ct-eyebrow">RELATED EVIDENCE</p><h2>Similarity & warnings</h2><p className="ct-report-summary">{result.similarity?.matches?.length ?? 0} reference compounds matched. {result.similarity?.label_summary?.high_risk_labels ?? 0} high-risk labels reported.</p>{result.warnings?.length ? result.warnings.map((warning) => <p className="ct-warning-row" key={warning}>{warning.replaceAll("_", " ")}</p>) : <p className="ct-report-detail">No pipeline warnings were returned.</p>}</article>
          </section>
          <section className="ct-panel ct-report-artifacts">
            <div className="ct-panel-heading"><div><p className="ct-eyebrow">GENERATED OUTPUTS</p><h2>Model artifacts</h2></div></div>
            <div className="ct-image-strip">
              {artifactUrl(result, "dosage", "dose_response_curves") && <figure><img src={artifactUrl(result, "dosage", "dose_response_curves")} alt="Dose-response curves for the three ion channels" /><figcaption>Ion-channel dose response</figcaption></figure>}
              {artifactUrl(result, "ord", "voltage_plot") && <figure><img src={artifactUrl(result, "ord", "voltage_plot")} alt="ORd action-potential voltage trace" /><figcaption>ORd action-potential trace</figcaption></figure>}
              {artifactUrl(result, "classification_xai", "bar_plot") && <figure><img src={artifactUrl(result, "classification_xai", "bar_plot")} alt="Classifier SHAP feature importance" /><figcaption>Classifier feature importance</figcaption></figure>}
            </div>
          </section>
          <div className="ct-page-actions"><Link className="ct-float-button ct-button-light" to="/xai"><ArrowLeft size={16} /> Back to evidence</Link><button className="ct-float-button" onClick={download}><Download size={16} /> Download JSON report</button><button className="ct-icon-button" onClick={() => setResult(null)} title="Clear current result" aria-label="Clear current result"><RotateCcw size={17} /></button></div>
          <p className="ct-disclaimer">Computational research output. This report is not a clinical decision tool.</p>
        </>
      )}
    </div>
  );
}