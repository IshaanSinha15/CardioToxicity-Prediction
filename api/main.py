from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from pipeline.utils import PipelineError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "classification_backend" / "inference" / "results"
ALLOWED_ARTIFACTS = {".png", ".svg"}

app = FastAPI(title="CardioTox Pipeline API", version="1.0.0")
origins = [
    origin.strip()
    for origin in os.getenv(
        "CARDIOTOX_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=1000)
    concentration_nm: float | None = Field(default=None, gt=0, le=100000)
    dose_nm: float | None = Field(default=None, gt=0, le=100000)
    drug_name: str | None = None
    skip_simulation: bool = False

    @model_validator(mode="after")
    def require_concentration(self) -> "PredictionRequest":
        if self.concentration_nm is None and self.dose_nm is None:
            raise ValueError("Provide concentration_nm or dose_nm.")
        return self


def _artifact_url(path: Any) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(RESULTS_DIR.resolve())
    except ValueError:
        return None
    if candidate.suffix.lower() not in ALLOWED_ARTIFACTS or not candidate.is_file():
        return None
    return f"/api/artifacts/{candidate.name}"


def _artifact_urls(artifacts: dict[str, Any]) -> dict[str, dict[str, str]]:
    urls: dict[str, dict[str, str]] = {}
    for group, paths in artifacts.items():
        if not isinstance(paths, dict):
            continue
        mapped = {
            name: url
            for name, path in paths.items()
            if (url := _artifact_url(path)) is not None
        }
        urls[group] = mapped
    return urls


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "cardiotox-pipeline"}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    from fastapi.encoders import jsonable_encoder

    concentration = request.concentration_nm or request.dose_nm
    payload: dict[str, Any] = {
        "smiles": request.smiles.strip(),
        "concentration_nm": concentration,
        "drug_name": request.drug_name,
    }
    if request.skip_simulation:
        payload["skip_simulation"] = True

    try:
        from pipeline.prediction_pipeline import PredictionPipeline

        result = PredictionPipeline().run(payload)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ImportError as exc:
        dependency = exc.name or str(exc)
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline dependency unavailable: {dependency}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    response = dict(result)
    response["artifact_urls"] = _artifact_urls(result.get("artifacts", {}))
    return jsonable_encoder(response)


@app.get("/api/artifacts/{filename}")
def artifact(filename: str) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = (RESULTS_DIR / filename).resolve()
    try:
        path.relative_to(RESULTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc
    if path.suffix.lower() not in ALLOWED_ARTIFACTS or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path)