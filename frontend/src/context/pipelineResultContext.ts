import { createContext } from "react";
import type { PipelineResult } from "../../services/predictionService";

interface PipelineResultContextValue {
  result: PipelineResult | null;
  setResult: (result: PipelineResult | null) => void;
}

export const PipelineResultContext = createContext<PipelineResultContextValue | null>(null);
export const STORAGE_KEY = "cardiotox_pipeline_result";

export function readSavedResult(): PipelineResult | null {
  try {
    const saved = sessionStorage.getItem(STORAGE_KEY);
    return saved ? (JSON.parse(saved) as PipelineResult) : null;
  } catch {
    return null;
  }
}