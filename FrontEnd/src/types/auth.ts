export interface SignUpRequest {
  username: string
  email: string
  password: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface UserPublic {
  id: number
  username: string
  email: string
  created_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  user: UserPublic
}

export interface GalleryPerson {
  id: number
  person_name: string
  image_path: string
  created_at: string
}

export interface GalleryListResponse {
  count: number
  items: GalleryPerson[]
}

export interface KnownDetection {
  status: 'known'
  person_name: string
  confidence: number
}

export interface UnknownDetection {
  status: 'unknown'
  confidence: number
  snapshot_path: string
  unknown_id: number
}

export type FaceDetection = KnownDetection | UnknownDetection

export interface MonitorFrameResponse {
  detections: FaceDetection[]
  threshold: number
  faces_detected: number
}

export interface RecognitionEvent {
  id: number
  person_name: string
  confidence: number
  snapshot_path: string | null
  timestamp: string
}

export interface RecognitionListResponse {
  count: number
  items: RecognitionEvent[]
}

export interface UnknownPersonRecord {
  id: number
  snapshot_path: string
  confidence: number
  timestamp: string
  action_taken: string | null
}

export interface UnknownListResponse {
  count: number
  items: UnknownPersonRecord[]
}

export interface SecurityEvent {
  id: number
  event_type: string
  description: string
  timestamp: string
}

export interface SecurityEventListResponse {
  count: number
  items: SecurityEvent[]
}

export interface AddToGalleryRequest {
  unknown_id: number
  person_name: string
}

export interface UnknownActionRequest {
  unknown_id: number
}

export interface ActionMessageResponse {
  message: string
}
