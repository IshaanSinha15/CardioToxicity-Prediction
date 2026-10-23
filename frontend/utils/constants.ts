export const APP_NAME = "CardioTox AI";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "http://localhost:8000";

export const API_TIMEOUT = 30000;

export const MAX_SMILES_LENGTH = 1000;

export const MIN_DOSE = 0.1;

export const MAX_DOSE = 100000;

export const STORAGE_KEYS = {
  HISTORY: "cardiotox_history",
};

export const CHANNELS = [
  "IKr",
  "INa",
  "ICaL",
] as const;

export const ROUTES = {
  HOME: "/",
  DASHBOARD: "/dashboard",
  PREDICTION: "/prediction",
  HISTORY: "/history",
  ABOUT: "/about",
  DOCUMENTATION: "/documentation",
};

export const COLORS = {
  PRIMARY: "#2563EB",
  SECONDARY: "#14B8A6",
  SUCCESS: "#22C55E",
  WARNING: "#F59E0B",
  ERROR: "#EF4444",
};