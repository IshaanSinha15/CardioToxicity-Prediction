import { ArrowLeft, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { usePipelineResult } from "../context/usePipelineResult";

interface ChemicalFragment { name?: string; importance?: number; message?: string; substructure?: string }
interface FeatureImpact { feature?: string; name?: string; importance?: number; value?: number; shap_value?: number }

function image(path?: string) {
  return path ? `${import.meta.env.VITE_API_BASE_URL ?? ""}${path}` : undefined;
}

function valueText(value: unknown) {
  return typeof value === "string" || typeof value === "number" ? String(value) : "Not reported";
}

export default function XAIDashboard() {
  const { result } = usePipelineResult();
  const chemical = result?.xai?.chemical?.result ?? {};
  const classifier = result?.xai?.classification?.result ?? {};
  const fragments = (chemical.substructures as ChemicalFragment[] | undefined) ?? [];
  const explanations = (chemical.explanation as ChemicalFragment[] | undefined) ?? [];
  const featureImpacts = (classifier.top_features as FeatureImpact[] | undefined) ?? [];
  const moleculeSvg = image(result?.artifact_urls?.chemical_xai?.molecule_svg);
  const shapBar = image(result?.artifact_urls?.classification_xai?.bar_plot);
  const shapWaterfall = image(result?.artifact_urls?.classification_xai?.waterfall_plot);
  const matches = result?.similarity?.matches ?? [];

  return (
    <div className="ct-page">
      <div className="ct-page-heading"><div><p className="ct-eyebrow">03 / EXPLAINABILITY</p><h1>Evidence & similarity</h1><p className="ct-lede">Inspect flagged chemical fragments, classifier feature effects and nearest reference compounds.</p></div><span className="ct-heading-mark">03<span>—</span>04</span></div>
      {!result ? (
        <section className="ct-panel ct-empty-dashboard"><h2>No analysis to explain yet</h2><p>Run a molecule through regression first. Chemical and classifier explanations are attached to that run.</p><Link className="ct-float-button" to="/prediction">Go to regression <ArrowRight size={16} /></Link></section>
      ) : (
        <>
          <section className="ct-xai-grid">
            <article className="ct-panel ct-xai-panel">
              <div className="ct-panel-heading"><div><p className="ct-eyebrow">CHEMICAL XAI</p><h2>Structure signals</h2></div><span className="ct-status">{result.xai?.chemical?.status ?? "unavailable"}</span></div>
              {moleculeSvg ? <img className="ct-molecule-svg" src={moleculeSvg} alt="Molecule with highlighted explanatory substructures" /> : <p className="ct-muted">Highlighted structure unavailable for this run.</p>}
              <div className="ct-explanation-list">
                {fragments.length ? fragments.map((item, index) => <div className="ct-explanation-row" key={`${item.name ?? "fragment"}-${index}`}><span className="ct-explanation-marker" /><div><strong>{item.name ?? item.substructure ?? `Fragment ${index + 1}`}</strong><p>{explanations[index]?.message ?? "Detected substructure considered by the chemical explanation model."}</p></div><b>{typeof item.importance === "number" ? item.importance.toFixed(2) : "--"}</b></div>) : <p className="ct-muted">{result.xai?.chemical?.error ?? "No explanatory fragments were returned."}</p>}
              </div>
            </article>
            <article className="ct-panel ct-xai-panel">
              <div className="ct-panel-heading"><div><p className="ct-eyebrow">CLASSIFICATION XAI</p><h2>Model feature effects</h2></div><span className="ct-status">{result.xai?.classification?.status ?? "unavailable"}</span></div>
              <div className="ct-classification-result"><span>Predicted class</span><strong>{valueText(result.classification?.predicted_class)}</strong><small>Confidence {valueText(classifier.confidence)}</small></div>
              {shapBar && <img className="ct-result-image ct-xai-image" src={shapBar} alt="SHAP feature importance bar plot" />}
              {shapWaterfall && <img className="ct-result-image ct-xai-image" src={shapWaterfall} alt="SHAP waterfall explanation" />}
              {featureImpacts.length > 0 && <div className="ct-feature-list">{featureImpacts.slice(0, 8).map((item, index) => <div className="ct-feature-row" key={`${item.feature ?? item.name ?? "feature"}-${index}`}><span>{item.feature ?? item.name ?? `Feature ${index + 1}`}</span><strong>{typeof item.importance === "number" ? item.importance.toFixed(3) : typeof item.value === "number" ? item.value.toFixed(3) : typeof item.shap_value === "number" ? item.shap_value.toFixed(3) : "--"}</strong></div>)}</div>}
              {!shapBar && !shapWaterfall && !featureImpacts.length && <p className="ct-muted">{result.xai?.classification?.error ?? "Classifier explanation requires a completed ORd simulation."}</p>}
            </article>
          </section>

          <section className="ct-panel ct-similarity-panel">
            <div className="ct-panel-heading"><div><p className="ct-eyebrow">REFERENCE SEARCH</p><h2>Nearest compounds</h2><p className="ct-muted">{result.similarity?.method ?? "Morgan fingerprint similarity"}</p></div><span className="ct-sim-count">{matches.length} matches</span></div>
            {matches.length ? <div className="ct-table-wrap"><table className="ct-table"><thead><tr><th>Reference</th><th>SMILES</th><th>Similarity</th><th>Risk label</th></tr></thead><tbody>{matches.map((match, index) => <tr key={`${valueText(match.reference_id)}-${index}`}><td>{valueText(match.name_or_id ?? match.reference_id)}</td><td><code>{valueText(match.original_smiles)}</code></td><td>{typeof match.similarity === "number" ? `${(match.similarity * 100).toFixed(1)}%` : valueText(match.similarity)}</td><td>{valueText(match.risk_label)}</td></tr>)}</tbody></table></div> : <div className="ct-empty-inline">{result.similarity?.status === "reference_database_unavailable" ? "Similarity reference data is not available." : "No compounds met the similarity threshold."}</div>}
          </section>
          <div className="ct-page-actions"><Link className="ct-float-button ct-button-light" to="/ord"><ArrowLeft size={16} /> Back to simulation</Link><Link className="ct-float-button" to="/report">Open combined report <ArrowRight size={16} /></Link></div>
        </>
      )}
    </div>
  );
}