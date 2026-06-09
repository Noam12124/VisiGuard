/** GET / — read_root() */
export interface RootResponse {
  message: string
}

/** POST /predict — predict() */
export interface PredictResponse {
  status: string
}

/** POST /compare-faces */
export interface CompareFacesResponse {
  is_match: boolean
  similarity_score: number
  threshold: number
  match_percent: number
  identity_label: 'Match' | 'No Match'
}

/** GET /video-feed/status */
export interface VideoFeedStatusResponse {
  connected: boolean
  source_index: number
}

/** GET /cameras */
export interface CameraSourceItem {
  index: number
  label: string
  available: boolean
}

export interface CamerasListResponse {
  sources: CameraSourceItem[]
  active_index: number
}

/** POST /set-camera */
export interface SetCameraRequest {
  source_index: number
}

export interface SetCameraResponse {
  success: boolean
  source_index: number
  connected: boolean
  message: string
}

export type AlertSeverity = 'high' | 'medium' | 'low'

/** GET /alerts — AlertItem */
export interface AlertItem {
  id: string
  timestamp: string
  person_label: string
  confidence: number | null
  camera_id: string
  thumbnail_url: string | null
  severity: AlertSeverity
}

export interface AlertsResponse {
  items: AlertItem[]
}

/** GET /recognized — RecognizedItem */
export interface RecognizedItem {
  id: string
  timestamp: string
  person_name: string
  similarity_score: number
  camera_id: string
  status: 'authorized'
}

export interface RecognizedResponse {
  items: RecognizedItem[]
}

/** POST /analyze-frame */
export interface AnalyzeFrameResponse {
  status: 'authorized' | 'unknown'
  name: string | null
  similarity_score: number | null
  threshold: number
  embedding: number[] | null
}

/** POST /add-known-user */
export interface AddKnownUserRequest {
  name: string
  embedding: number[]
}

export interface AddKnownUserResponse {
  id: string
  name: string
  message: string
}

/** POST /report-intrusion */
export interface ReportIntrusionRequest {
  similarity_score?: number | null
}

export interface ReportIntrusionResponse {
  logged: boolean
  message: string
}

/** FastAPI HTTP error body */
export interface ApiErrorDetail {
  detail:
    | string
    | { loc: (string | number)[]; msg: string; type: string }[]
}

export type ApiSuccessResponse =
  | RootResponse
  | PredictResponse
  | CompareFacesResponse
  | VideoFeedStatusResponse
  | CamerasListResponse
  | SetCameraResponse
  | AlertsResponse
  | RecognizedResponse
  | AnalyzeFrameResponse
  | AddKnownUserResponse
  | ReportIntrusionResponse
