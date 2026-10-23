# CardioTox Prediction Workspace

The application runs the existing `pipeline.PredictionPipeline` from a React frontend. One submitted SMILES/concentration runs regression, dose-response, ORd simulation, chemical and classifier XAI, and similarity analysis; the returned result is shared by four dashboards.

## Run locally

Use Python 3.10 and Node.js 20.19+ (or 22.12+). The pinned `rdkit-pypi==2022.9.5` dependency targets Python 3.10; the workspace `.conda` environment is currently pinned to Python 3.15 and is not compatible with the model stack. Create or activate a compatible environment before starting the API:

```powershell
conda create -n cardiotox python=3.10 -y
conda activate cardiotox
python -m pip install -r requirements.txt
```

1. Install the Python dependencies from the repository root: `python -m pip install -r requirements.txt`.
2. Start the API from the repository root: `uvicorn api.main:app --reload --port 8000`.
3. In a second terminal, install frontend dependencies and start Vite: `cd frontend`, `npm install`, then `npm run dev`.
4. Open the URL printed by Vite, normally `http://localhost:5173`.

Vite proxies `/api` to `http://localhost:8000` during development. Set `VITE_API_BASE_URL` in the frontend environment if the API is hosted separately. Set `CARDIOTOX_CORS_ORIGINS` to a comma-separated list of allowed frontend origins for cross-origin deployments.

The full model pipeline also needs the model artifacts and scientific dependencies listed in `requirements.txt`. On Windows, Myokit's ORd simulation compiles a native model at runtime and requires Microsoft C++ Build Tools 14 or newer. When simulation is unavailable, the API still returns the other completed sections and includes a warning/status; classifier XAI requires completed ORd features.

## API

- `GET /api/health` reports whether the HTTP service is responding.
- `POST /api/predict` accepts `{"smiles":"CCO","concentration_nm":100}`. `dose_nm` is also accepted for existing callers; the concentration must be greater than zero and at most 100,000 nM.
- The prediction response contains the pipeline result and `artifact_urls` for generated PNG/SVG files.
- `GET /api/artifacts/{filename}` serves generated images from `classification_backend/inference/results`; file access is restricted to PNG/SVG outputs in that directory.

The four dashboards are `/prediction` (regression and dose response), `/ord` (cell simulation), `/xai` (chemical/classifier explanations and similarity), and `/report` (combined summary). Results survive route changes and page refreshes within the browser session.

## Frontend checks

From `frontend/`, run `npm run build` and `npm run lint`.