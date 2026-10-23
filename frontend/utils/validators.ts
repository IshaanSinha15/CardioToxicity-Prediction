import {
  MAX_DOSE,
  MAX_SMILES_LENGTH,
  MIN_DOSE,
} from "./constants";

export function validateSmiles(
  smiles: string
): string | null {
  if (!smiles.trim()) {
    return "SMILES string is required.";
  }

  if (smiles.length > MAX_SMILES_LENGTH) {
    return `Maximum ${MAX_SMILES_LENGTH} characters allowed.`;
  }

  return null;
}

export function validateDose(
  dose: number
): string | null {
  if (Number.isNaN(dose)) {
    return "Dose must be a number.";
  }

  if (dose < MIN_DOSE) {
    return `Dose must be at least ${MIN_DOSE} nM.`;
  }

  if (dose > MAX_DOSE) {
    return `Dose cannot exceed ${MAX_DOSE} nM.`;
  }

  return null;
}

export function validatePredictionInput(
  smiles: string,
  dose: number
) {
  const smilesError =
    validateSmiles(smiles);

  if (smilesError) {
    return {
      valid: false,
      error: smilesError,
    };
  }

  const doseError =
    validateDose(dose);

  if (doseError) {
    return {
      valid: false,
      error: doseError,
    };
  }

  return {
    valid: true,
    error: null,
  };
}