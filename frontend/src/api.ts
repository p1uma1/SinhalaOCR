import type {
  CharacterOCRResult,
  DocumentOCRResult,
  HealthStatus,
  LineOCRResult,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(endpoint: string, file: File): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : "Something went wrong. Please try again.";
    throw new ApiError(detail, response.status);
  }

  return data as T;
}

export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) {
    throw new ApiError("API is unavailable", response.status);
  }
  return response.json();
}

export const ocrLine = (file: File) => request<LineOCRResult>("/api/ocr/line", file);
export const ocrDocument = (file: File) =>
  request<DocumentOCRResult>("/api/ocr/document", file);
export const ocrCharacter = (file: File) =>
  request<CharacterOCRResult>("/api/ocr/character", file);
