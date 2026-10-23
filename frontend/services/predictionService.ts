import api from "./api";

export interface PredictionRequest {
  smiles: string;
  concentration_nm: number;
}

export interface PipelineResult {
  input: {
    smiles: string;
    concentration_nm?: number;
    dose_nm?: number;
    drug_name?: string | null;
  };
  ic50_prediction?: Record<string, { IC50_nM?: number; pIC50?: number }>;
  dose_response?: Record<string, number>;
  simulation?: {
    status?: string;
    features?: Record<string, number>;
    error?: string;
  };
  similarity?: {
    status?: string;
    method?: string;
    matches?: Array<Record<string, string | number | null>>;
    label_summary?: Record<string, boolean | number>;
  };
  interpretation?: Record<string, unknown>;
  mechanistic_classification?: {
    channels?: Record<string, { band?: string; block_pct?: number }>;
    overall_level?: string;
    dominant_channel?: string | null;
  };
  frontend?: {
    interpretation?: { summary?: string; details?: string[] };
  };
  mechanistic_risk?: Record<string, unknown>;
  classification?: {
    predicted_class?: number | null;
    probabilities?: number[] | Record<string, number>;
  };
  xai?: {
    chemical?: { status?: string; result?: Record<string, unknown>; error?: string };
    classification?: { status?: string; result?: Record<string, unknown>; error?: string };
  };
  artifacts?: Record<string, Record<string, string>>;
  artifact_urls?: Record<string, Record<string, string>>;
  warnings?: string[];
}

class PredictionService {
  async predict(
    data: PredictionRequest
  ): Promise<PipelineResult> {
    const response = await api.post(
      "/api/predict",
      data
    );

    return response.data;
  }

  async health() {
    const response = await api.get("/api/health");
    return response.data;
  }
}

export default new PredictionService();