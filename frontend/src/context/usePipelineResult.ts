import { useContext } from "react";
import { PipelineResultContext } from "./pipelineResultContext";

export function usePipelineResult() {
  const context = useContext(PipelineResultContext);
  if (!context) {
    throw new Error("usePipelineResult must be used within PredictionProvider.");
  }
  return context;
}

export function usePredictionContext() {
  const { result, setResult } = usePipelineResult();
  return { prediction: result, setPrediction: setResult };
}