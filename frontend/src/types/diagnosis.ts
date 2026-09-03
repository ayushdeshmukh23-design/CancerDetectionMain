export type DiagnosisPrediction = "Normal" | "Benign" | "Malignant" | "Unknown"
export type RiskLevel = "Low" | "Moderate" | "High" | "Unknown"

export interface EnsembleResult {
  prediction: DiagnosisPrediction
  confidence: number
  reliability: number
  risk_level: RiskLevel
  risk_score: number
  birads?: string
  tnm_stage_est?: string
  probabilities: Record<string, number>
  model_contributions?: Record<string, Record<string, number>>
  contributions?: Record<string, Record<string, number>>
  thermal_delta_t?: number
  model_agreement?: number
  xai?: {
    top_features: string[]
  }
  explanation?: string
  reasoning_payload?: Record<string, any>
}

export interface PreprocessedImages {
  original?: string
  enhanced?: string
  roi?: string
  edges?: string
  spectral_fusion?: string
}

export interface PipelineLogEntry {
  timestamp?: string
  stage?: string
  message: string
  level?: "INFO" | "WARN" | "ERROR" | "AUDIT"
  latency_ms?: number
  device?: string
  tensor_shape?: string
  details?: Record<string, any>
}

export interface HistoricalPoint {
  idx: number
  date: string
  risk_score: number
  confidence: number
  reliability: number
  risk_level?: string
  birads?: string
  doubling_time_days?: number | null
  thermal_delta_t?: number
  tumor_volume_cm3?: number
  model_agreement?: number
}

export interface RadiomicsProfile {
  glcm_contrast?: number
  glcm_homogeneity?: number
  glcm_entropy?: number
  spiculation_index?: number
  compactness?: number
  perimeter_area_ratio?: number
  thermal_delta_t?: number
  vascular_flow_index?: number
}

export interface UncertaintyMetrics {
  epistemic: number
  aleatoric: number
  confidence_interval_95: [number, number]
  entropy: number
}

export interface ModelConcordance {
  cohens_kappa: number
  fleiss_kappa: number
  consensus_agreement_pct: number
}

export interface DoublingTimeAnalytics {
  doubling_time_days: number
  current_volume_cm3: number
  baseline_volume_cm3: number
  growth_rate_pct_month: number
  kinetic_model: string
}

export interface CalibrationMetrics {
  brier_score: number
  ece_pct: number
  calibration_slope: number
}

export interface AgentVisionOutputs {
  input_image?: string
  stage1_segmentation?: string
  stage1_roi?: string
  stage2_texture?: string
  stage2_deep_cnn?: string
  stage2_overlay?: string
  stage3_gradcam?: string
  stage3_lime?: string
  stage3_shap?: string
}

export interface DiagnosisState {
  case_id?: string
  birads?: string
  image_path?: string | null
  thermal_matrix_path?: string | null
  patient_context?: string
  preprocessed_image?: PreprocessedImages | null
  segmentation_mask?: string | null
  ensemble_result?: EnsembleResult | null
  gradcam_heatmap?: string | null
  thermal_heatmap?: string | null
  thermal_features?: Record<string, any>
  lime_explanation?: string | null
  vit_attention_map?: string | null
  integrated_gradients_map?: string | null
  spectral_fusion_map?: string | null
  edges_map?: string | null
  agent_vision_outputs?: AgentVisionOutputs
  shap_values?: Record<string, number>
  radiomics_profile?: RadiomicsProfile
  uncertainty?: UncertaintyMetrics
  model_concordance?: ModelConcordance
  doubling_time_analytics?: DoublingTimeAnalytics
  calibration_metrics?: CalibrationMetrics
  markdown_report?: string | null
  llm_response?: string | null
  pipeline_logs?: (string | PipelineLogEntry)[]
  processing_time?: number
  seeded_history?: HistoricalPoint[]
}

export interface SystemStatus {
  models_loaded: boolean
  model_details: Record<string, boolean>
  llm_connected: boolean
  llm_model: string
  timestamp: number
}

export interface ImageUploadResponse {
  success: boolean
  image_path: string
  filename: string
  preview_url: string
  width: number
  height: number
  format: string
}

export interface ThermalUploadResponse {
  success: boolean
  thermal_path: string
  filename: string
  matrix: number[][]
  shape: number[]
  min_val: number
  max_val: number
  mean_val: number
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
}
