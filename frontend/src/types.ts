export type TabId = "line" | "document" | "character";

export interface HealthStatus {
  device: string;
  cuda_available: boolean;
  stage1_ready: boolean;
  stage2_ready: boolean;
  num_classes: number;
}

export interface LineOCRResult {
  text: string;
}

export interface CharacterPrediction {
  label: string;
  confidence: number;
}

export interface CharacterOCRResult {
  top_prediction: CharacterPrediction;
  predictions: CharacterPrediction[];
}

export interface DocumentLine {
  index: number;
  image_base64: string;
  text: string;
}

export interface DocumentOCRResult {
  line_count: number;
  lines: DocumentLine[];
  full_text: string;
  debug?: {
    original?: string;
    deskewed?: string;
  };
}
