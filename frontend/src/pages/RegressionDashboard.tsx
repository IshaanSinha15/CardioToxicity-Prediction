import { useState, type FormEvent } from "react";
import { ArrowDownRight, ArrowRight, LoaderCircle, RotateCcw } from "lucide-react";
import { Link } from "react-router-dom";
import usePrediction from "../../hooks/usePrediction";
import type { PipelineResult } from "../../services/predictionService";

const channels = [
  { key: "herg", block: "herg_block", name: "hERG", sub: "IKr", color: "coral" },
  { key: "nav", block: "nav_block", name: "NaV1.5", sub: "INa", color: "blue" },
  { key: "cav", block: "cav_block", name: "CaV1.2", sub: "ICaL", color: "green" },
] as const;

function display(value: number | undefined, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "--";
}

function imageUrl(result: PipelineResult, key: string) {
  const path = result.artifact_urls?.dosage?.[key];
  return path ? `${import.meta.env.VITE_API_BASE_URL ?? ""}${path}` : undefined;
}

export default function RegressionDashboard() {
  const { result, loading, error, predict } = usePrediction();
  const [smiles, setSmiles] = useState(result?.input.smiles ?? "");
  const [concentration, setConcentration] = useState(
    String(result?.input.concentration_nm ?? result?.input.dose_nm ?? 750),
  );
  const [formError, setFormError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const amount = Number(concentration);
    if (!smiles.trim()) {
      setFormError("Enter a SMILES string to start the analysis.");
      return;
    }
    if (!Number.isFinite(amount) || amount <= 0) {
      setFormError("Concentration must be a number greater than zero.");
      return;
    }
    setFormError("");
    await predict(smiles.trim(), amount);
  }

  function reset() {
    setSmiles("");
    setConcentration("750");
    setFormError("");
  }

  return (
    <div className="ct-page">
      <div className="ct-page-heading">
        <div>
          <p className="ct-eyebrow">01 / REGRESSION</p>
          <h1>Ion-channel response</h1>
          <p className="ct-lede">Run the molecular pipeline and inspect predicted potency and concentration-dependent block.</p>
        </div>
        <span className="ct-heading-mark">01<span>—</span>04</span>
      </div>

      <div className="ct-regression-grid">
        <form className="ct-panel ct-input-panel" onSubmit={submit}>
          <div className="ct-section-title"><span>01</span><div><h2>Molecular input</h2><p>SMILES and test concentration</p></div></div>
          <label className="ct-field-label" htmlFor="smiles-input">SMILES</label>
          <textarea
            id="smiles-input"
            className="ct-textarea"
            value={smiles}
            onChange={(event) => setSmiles(event.target.value)}
            placeholder="Paste a molecular SMILES string"
            spellCheck={false}
            maxLength={1000}
            required
          />
          <button className="ct-text-action" type="button" onClick={() => setSmiles("CCOC(=O)c1ccccc1")}>Load example <ArrowDownRight size={14} /></button>
          <label className="ct-field-label" htmlFor="concentration-input">CONCENTRATION <span>nM</span></label>
          <div className="ct-dose-control">
            <input id="concentration-input" type="number" min="0.1" max="100000" step="any" value={concentration} onChange={(event) => setConcentration(event.target.value)} required />
            <input aria-label="Concentration slider" type="range" min="0.1" max="10000" step="10" value={Math.min(Number(concentration) || 0.1, 10000)} onChange={(event) => setConcentration(event.target.value)} />
          </div>
          <div className="ct-range-labels"><span>0.1 nM</span><span>10,000 nM</span></div>
          {(formError || error) && <p className="ct-error" role="alert">{formError || error}</p>}
          <div className="ct-form-actions">
            <button className="ct-float-button" type="submit" disabled={loading}>
              {loading ? <><LoaderCircle className="ct-spin" size={17} /> Running pipeline</> : <>Run full analysis <ArrowRight size={17} /></>}
            </button>
            <button className="ct-icon-button" type="button" onClick={reset} aria-label="Reset form" title="Reset form"><RotateCcw size={17} /></button>
          </div>
          {loading && <p className="ct-loading-note">Regression, dose-response, ORd simulation and explanations are being calculated.</p>}
        </form>

        <section className="ct-panel ct-results-panel" aria-live="polite">
          <div className="ct-section-title"><span>02</span><div><h2>Potency by channel</h2><p>{result ? `${result.input.concentration_nm ?? result.input.dose_nm ?? "--"} nM test concentration` : "Results appear after a pipeline run"}</p></div></div>
          {result ? (
            <>
              <div className="ct-channel-list">
                {channels.map((channel) => {
                  const ic50 = result.ic50_prediction?.[channel.key]?.IC50_nM;
                  const block = result.dose_response?.[channel.block];
                  const pct = Math.max(0, Math.min(100, (block ?? 0) <= 1 ? (block ?? 0) * 100 : (block ?? 0)));
                  return (
                    <article className={`ct-channel ct-${channel.color}`} key={channel.key}>
                      <div className="ct-channel-name"><span className="ct-channel-dot" /><div><strong>{channel.name}</strong><small>{channel.sub}</small></div></div>
                      <div className="ct-channel-potency"><strong>{display(ic50)}</strong><small>nM IC50</small></div>
                      <div className="ct-channel-block"><div><span>Predicted block</span><strong>{display(pct, 1)}%</strong></div><div className="ct-meter"><i style={{ width: `${pct}%` }} /></div></div>
                    </article>
                  );
                })}
              </div>
              <div className="ct-result-footnote">Predictions are computational estimates for research use; validate against experimental evidence.</div>
            </>
          ) : (
            <div className="ct-empty-state"><span className="ct-empty-glyph">↗</span><p>Submit a molecule to reveal three-channel IC50 estimates and block percentages.</p></div>
          )}
        </section>
      </div>

      {result && (
        <section className="ct-panel ct-chart-panel">
          <div className="ct-panel-heading"><div><p className="ct-eyebrow">DOSE RESPONSE</p><h2>Concentration vs. predicted block</h2></div><Link className="ct-inline-link" to="/ord">Continue to ORd <ArrowRight size={15} /></Link></div>
          {imageUrl(result, "dose_response_curves") ? <img className="ct-result-image" src={imageUrl(result, "dose_response_curves")} alt="Dose-response curves for hERG, NaV1.5 and CaV1.2" /> : <p className="ct-muted">Dose-response plot was not generated for this run.</p>}
          <div className="ct-image-strip">
            {imageUrl(result, "channel_block") && <figure><img src={imageUrl(result, "channel_block")} alt="Channel block percentages" /><figcaption>Channel block at submitted concentration</figcaption></figure>}
            {imageUrl(result, "safety_margin") && <figure><img src={imageUrl(result, "safety_margin")} alt="Safety margins by channel" /><figcaption>Predicted safety margins</figcaption></figure>}
          </div>
        </section>
      )}
    </div>
  );
}