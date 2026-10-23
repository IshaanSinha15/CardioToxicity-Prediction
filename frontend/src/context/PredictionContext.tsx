import { useState, type ReactNode } from "react";
import { PipelineResultContext, readSavedResult, STORAGE_KEY } from "./pipelineResultContext";
import type { PipelineResult } from "../../services/predictionService";

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [result, setResultState] = useState<PipelineResult | null>(readSavedResult);

  const setResult = (nextResult: PipelineResult | null) => {
    setResultState(nextResult);
    try {
      if (nextResult) {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(nextResult));
      } else {
        sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      return;
    }
  };

  return (
    <PipelineResultContext.Provider value={{ result, setResult }}>
      {children}
    </PipelineResultContext.Provider>
  );
}
