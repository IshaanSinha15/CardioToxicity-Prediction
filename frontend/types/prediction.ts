export interface IC50Prediction {
  IC50_nM: number;
}

export interface DoseResponse {
  herg_block: number;
  nav_block: number;
  cav_block: number;
}

export interface PredictionInput {
  smiles: string;
  dose_nm: number;
}

export interface PredictionResult {
  input: PredictionInput;

  ic50_prediction: {
    herg: IC50Prediction;
    nav: IC50Prediction;
    cav: IC50Prediction;
  };

  dose_response: DoseResponse;
}

export interface PredictionHistoryItem {
  id: number;
  smiles: string;
  dose: number;
  date: string;
}