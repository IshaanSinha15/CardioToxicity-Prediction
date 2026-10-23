import {
  AlertCircle,
  ArrowRight,
  Beaker,
  CheckCircle2,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";

import {
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
} from "react";

import usePrediction, {
  type PredictionResult,
} from "../../hooks/usePrediction";

import moleculeService from "../../services/moleculeService";


function formatNumber(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) {
    return "--";
  }

  return value.toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
}


function percent(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) {
    return 0;
  }

  const n = value <= 1 ? value * 100 : value;

  return Math.max(0, Math.min(100, n));
}


function riskLabel(value: number | undefined) {
  const p = percent(value);

  if (p >= 75) return "High";
  if (p >= 40) return "Moderate";
  if (p > 0) return "Low";

  return "No signal";
}


function highestResponse(
  result: PredictionResult | null
) {
  if (!result) {
    return 0;
  }

  return Math.max(
    percent(result.dose_response?.herg_block),
    percent(result.dose_response?.nav_block),
    percent(result.dose_response?.cav_block)
  );
}


/* =========================================================
   MOLECULE PREVIEW
   ========================================================= */

function MoleculePreview({
  smiles,
}: {
  smiles: string;
}) {
  const [svg, setSvg] = useState("");
  const [png, setPng] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function renderMolecule() {
      if (!smiles.trim()) {
        setSvg("");
        setPng("");
        return;
      }

      setLoading(true);

      try {
        const response =
          await moleculeService.renderMolecule(
            smiles.trim()
          );

        if (!cancelled) {
          setSvg(response?.svg || "");
          setPng(response?.png || "");
        }
      } catch {
        if (!cancelled) {
          setSvg("");
          setPng("");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    const timer = window.setTimeout(
      renderMolecule,
      350
    );

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [smiles]);


  if (!smiles.trim()) {
    return (
      <div className="research-molecule-empty">
        <Beaker size={28} />

        <span>
          Enter a SMILES string to view the
          molecular structure.
        </span>
      </div>
    );
  }


  if (loading) {
    return (
      <div className="research-molecule-empty">
        Rendering molecular structure...
      </div>
    );
  }


  if (svg) {
    return (
      <div
        className="research-molecule-svg"
        dangerouslySetInnerHTML={{
          __html: svg,
        }}
      />
    );
  }


  if (png) {
    return (
      <img
        className="research-molecule-image"
        src={png}
        alt="Rendered molecular structure"
      />
    );
  }


  return (
    <div className="research-molecule-fallback">
      <span>SMILES</span>

      <code>{smiles}</code>

      <small>
        Molecular rendering will appear when
        the molecule service is available.
      </small>
    </div>
  );
}


/* =========================================================
   CHANNEL RESULT CARD
   ========================================================= */

interface ChannelCardProps {
  name: string;
  label: string;
  ic50: number | undefined;
  response: number | undefined;
}


function ChannelCard({
  name,
  label,
  ic50,
  response,
}: ChannelCardProps) {
  const p = percent(response);
  const risk = riskLabel(response);

  return (
    <article className="research-channel-card">

      <div className="research-channel-header">

        <div>
          <strong>{name}</strong>

          <span>{label}</span>
        </div>

        <span
          className={`research-risk research-risk-${risk
            .toLowerCase()
            .replace(" ", "-")}`}
        >
          {risk}
        </span>

      </div>


      <div className="research-channel-value">

        <span>
          {formatNumber(ic50)}
        </span>

        <small>
          nM IC50
        </small>

      </div>


      <div className="research-meter">

        <span
          style={{
            width: `${p}%`,
          }}
        />

      </div>


      <div className="research-channel-footer">

        <span>
          Dose response
        </span>

        <span>
          {p.toFixed(0)}%
        </span>

      </div>

    </article>
  );
}


/* =========================================================
   PREDICTION PAGE
   ========================================================= */

export default function PredictionPage() {

  const [smiles, setSmiles] =
    useState("");

  const [dose, setDose] =
    useState("750");

  const [validationError, setValidationError] =
    useState("");


  const {
    loading,
    result,
    error: apiError,
    predict,
  } = usePrediction();


  const predictionResult =
    result as PredictionResult | null;


  const error =
    validationError || apiError;


  const overallRisk =
    useMemo(() => {

      const highest =
        highestResponse(
          predictionResult
        );

      if (highest >= 75) {
        return "High";
      }

      if (highest >= 40) {
        return "Moderate";
      }

      if (highest > 0) {
        return "Low";
      }

      return "Not available";

    }, [predictionResult]);


  /* =======================================================
     INPUT
     ======================================================= */

  const handleSmilesChange = (
    event: ChangeEvent<HTMLTextAreaElement>
  ) => {

    setSmiles(
      event.target.value
    );

    setValidationError("");
  };


  const handleDoseChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {

    setDose(
      event.target.value
    );
  };


  /* =======================================================
     EXAMPLE
     ======================================================= */

  const loadExample = () => {

    setSmiles(
      "CCOC(=O)c1ccccc1"
    );

    setValidationError("");
  };


  /* =======================================================
     RESET
     ======================================================= */

  const handleReset = () => {

    setSmiles("");
    setDose("750");
    setValidationError("");
  };


  /* =======================================================
     PREDICTION
     ======================================================= */

  const handlePrediction =
    async () => {

      if (!smiles.trim()) {

        setValidationError(
          "Enter a SMILES string before running the prediction."
        );

        return;
      }


      const numericDose =
        Number(dose);


      if (
        !dose ||
        Number.isNaN(numericDose) ||
        numericDose < 0
      ) {

        setValidationError(
          "Enter a valid experimental dose."
        );

        return;
      }


      setValidationError("");

      await predict(
        smiles.trim(),
        numericDose
      );
    };


  return (
    <div className="research-page">

      {/* =================================================
          HEADER
         ================================================= */}

      <section className="research-header">

        <div>

          <p className="research-kicker">
            CARDIOTOX PREDICTION
          </p>

          <h1>
            Cardiotoxicity Prediction
          </h1>

          <p>
            Estimate cardiac ion-channel
            response from molecular structure
            and experimental dose.
          </p>

        </div>


        <div className="research-header-status">

          <span className="research-online-dot" />

          <div>

            <strong>
              CardioTox-v1
            </strong>

            <small>
              Prediction service ready
            </small>

          </div>

        </div>

      </section>


      {/* =================================================
          INPUT + STRUCTURE
         ================================================= */}

      <section className="research-workspace">

        {/* INPUT */}

        <article className="research-card">

          <div className="research-card-heading">

            <div className="research-step">
              1
            </div>

            <div>

              <h2>
                Molecular input
              </h2>

              <p>
                Enter a SMILES string and
                experimental dose.
              </p>

            </div>

          </div>


          <label
            className="research-label"
            htmlFor="smiles"
          >
            <span>
              SMILES
            </span>

            <span>
              Required
            </span>
          </label>


          <textarea
            id="smiles"
            className="research-textarea"
            value={smiles}
            onChange={handleSmilesChange}
            placeholder="Paste molecular SMILES..."
            spellCheck={false}
          />


          <button
            type="button"
            className="research-example"
            onClick={loadExample}
          >
            Try example:{" "}
            <code>
              CCOC(=O)c1ccccc1
            </code>
          </button>


          <label
            className="research-label"
            htmlFor="dose"
          >
            <span>
              Experimental dose
            </span>

            <span>
              nM
            </span>
          </label>


          <div className="research-dose-row">

            <input
              id="dose"
              className="research-dose-input"
              type="number"
              min="0"
              max="10000"
              value={dose}
              onChange={handleDoseChange}
            />


            <input
              className="research-dose-range"
              type="range"
              min="0"
              max="10000"
              step="10"
              value={
                Number(dose) || 0
              }
              onChange={handleDoseChange}
              aria-label="Experimental dose"
            />

          </div>


          <div className="research-range-labels">

            <span>
              0
            </span>

            <span>
              5,000
            </span>

            <span>
              10,000 nM
            </span>

          </div>


          {error && (
            <div className="research-error">

              <AlertCircle size={16} />

              <span>
                {error}
              </span>

            </div>
          )}


          <div className="research-action-row">

            <button
              type="button"
              className="research-primary-button"
              onClick={handlePrediction}
              disabled={loading}
            >

              {loading
                ? "Running prediction..."
                : "Run prediction"}

              {!loading && (
                <ArrowRight size={17} />
              )}

            </button>


            <button
              type="button"
              className="research-reset-button"
              onClick={handleReset}
              aria-label="Reset"
            >

              <RotateCcw size={17} />

            </button>

          </div>

        </article>


        {/* MOLECULAR STRUCTURE */}

        <article className="research-card">

          <div className="research-card-heading">

            <div className="research-step">
              2
            </div>

            <div>

              <h2>
                Molecular structure
              </h2>

              <p>
                2D representation generated
                from the submitted SMILES.
              </p>

            </div>

          </div>


          <MoleculePreview
            smiles={smiles}
          />


          <div className="research-structure-meta">

            <span>

              <span className="research-online-dot" />

              SMILES

            </span>


            <code>
              {smiles ||
                "No structure loaded"}
            </code>

          </div>

        </article>

      </section>


      {/* =================================================
          RESULTS
         ================================================= */}

      {predictionResult && (

        <section className="research-results">

          <div className="research-results-heading">

            <div>

              <p className="research-kicker">
                PREDICTION RESULT
              </p>

              <h2>
                Model output
              </h2>

              <p>
                Values below are returned
                by the CardioTox prediction
                service.
              </p>

            </div>


            <div
              className={`research-overall research-risk-${overallRisk
                .toLowerCase()
                .replace(" ", "-")}`}
            >

              <small>
                Dose-response risk
              </small>

              <strong>
                {overallRisk}
              </strong>

            </div>

          </div>


          {/* SUMMARY */}

          <div className="research-summary-grid">

            <div className="research-summary-item">

              <span>
                Input dose
              </span>

              <strong>
                {
                  predictionResult
                    .input
                    ?.dose_nm ??
                  dose
                }{" "}
                nM
              </strong>

            </div>


            <div className="research-summary-item">

              <span>
                Highest response
              </span>

              <strong>
                {highestResponse(
                  predictionResult
                ).toFixed(0)}
                %
              </strong>

            </div>


            <div className="research-summary-item">

              <span>
                Model
              </span>

              <strong>
                CardioTox-v1
              </strong>

            </div>

          </div>


          {/* CHANNELS */}

          <div className="research-channel-grid">

            <ChannelCard
              name="hERG"
              label="IKr"
              ic50={
                predictionResult
                  .ic50_prediction
                  ?.herg
                  ?.IC50_nM
              }
              response={
                predictionResult
                  .dose_response
                  ?.herg_block
              }
            />


            <ChannelCard
              name="Nav1.5"
              label="INa"
              ic50={
                predictionResult
                  .ic50_prediction
                  ?.nav
                  ?.IC50_nM
              }
              response={
                predictionResult
                  .dose_response
                  ?.nav_block
              }
            />


            <ChannelCard
              name="Cav1.2"
              label="ICaL"
              ic50={
                predictionResult
                  .ic50_prediction
                  ?.cav
                  ?.IC50_nM
              }
              response={
                predictionResult
                  .dose_response
                  ?.cav_block
              }
            />

          </div>


          <div className="research-note">

            <ShieldCheck size={17} />

            <span>
              Research output only.
              Predictions should be
              interpreted alongside
              experimental evidence.
            </span>

          </div>

        </section>
      )}


      {/* =================================================
          WORKFLOW
         ================================================= */}

      <section className="research-workflow">

        <div>

          <p className="research-kicker">
            WORKFLOW
          </p>

          <h2>
            From molecule to result
          </h2>

        </div>


        <div className="research-workflow-grid">

          <div>

            <strong>
              01
            </strong>

            <span>
              Input
            </span>

            <p>
              Provide SMILES and dose.
            </p>

          </div>


          <div>

            <strong>
              02
            </strong>

            <span>
              Predict
            </span>

            <p>
              Run the trained
              prediction pipeline.
            </p>

          </div>


          <div>

            <strong>
              03
            </strong>

            <span>
              Review
            </span>

            <p>
              Compare IC50 and
              channel response.
            </p>

          </div>

        </div>

      </section>


      <div className="research-footer-note">

        <CheckCircle2 size={15} />

        <span>
          CardioTox-v1 · Molecular
          descriptors + dose-response
          analysis
        </span>

      </div>

    </div>
  );
}