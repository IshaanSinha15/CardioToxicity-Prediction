export interface ApiResponse<T> {
  success: boolean;
  message?: string;
  data: T;
}

export interface ApiError {
  status: number;
  message: string;
  details?: unknown;
}

export interface HealthResponse {
  status: string;
  version?: string;
}

export interface LoadingState {
  loading: boolean;
  error: string | null;
}