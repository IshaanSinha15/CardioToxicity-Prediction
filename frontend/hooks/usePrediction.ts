import { useState } from "react";
import { isAxiosError } from "axios";
import predictionService from "../services/predictionService";
import { usePipelineResult } from "../src/context/usePipelineResult";

export type { PipelineResult as PredictionResult } from "../services/predictionService";

export default function usePrediction() {
  const [loading, setLoading] = useState(false);
  const { result, setResult } = usePipelineResult();

  const [error, setError] =
    useState<string | null>(null);

  const predict = async (
    smiles: string,
    dose: number
  ) => {
    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const response =
        await predictionService.predict({
          smiles,
          concentration_nm: dose,
        });

      setResult(response);

      return response;
    } catch (err) {
      const message = isAxiosError(err)
        ? err.response?.data?.detail ?? err.message
        : err instanceof Error
          ? err.message
          : "Pipeline execution failed.";
      setError(String(message));
      return null;
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    result,
    error,
    predict,
  };
}