import { request } from '@/services/api-client'
import type {
  AddKnownUserRequest,
  AddKnownUserResponse,
  AlertsResponse,
  AnalyzeFrameResponse,
  ApiErrorDetail,
  CamerasListResponse,
  CompareFacesResponse,
  PredictResponse,
  RecognizedResponse,
  ReportIntrusionRequest,
  ReportIntrusionResponse,
  RootResponse,
  SetCameraRequest,
  SetCameraResponse,
  VideoFeedStatusResponse,
} from '@/types/api'
import type {
  ActionMessageResponse,
  AddToGalleryRequest,
  GalleryListResponse,
  GalleryPerson,
  LoginRequest,
  MonitorFrameResponse,
  RecognitionListResponse,
  SecurityEventListResponse,
  SignUpRequest,
  TokenResponse,
  UnknownActionRequest,
  UnknownListResponse,
} from '@/types/auth'

export { ApiError, getApiBaseUrl, getUploadUrl, getVideoFeedUrl } from '@/services/api-client'

/** GET / */
export function getHealth(): Promise<RootResponse> {
  return request<RootResponse>('/')
}

/** POST /predict */
export function postPredict(): Promise<PredictResponse> {
  return request<PredictResponse>('/predict', { method: 'POST' })
}

/** POST /compare-faces */
export function postCompareFaces(formData: FormData): Promise<CompareFacesResponse> {
  return request<CompareFacesResponse>('/compare-faces', {
    method: 'POST',
    data: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

/** GET /video-feed/status */
export function getVideoFeedStatus(): Promise<VideoFeedStatusResponse> {
  return request<VideoFeedStatusResponse>('/video-feed/status')
}

/** GET /cameras */
export function getCameras(): Promise<CamerasListResponse> {
  return request<CamerasListResponse>('/cameras')
}

/** POST /set-camera */
export function postSetCamera(body: SetCameraRequest): Promise<SetCameraResponse> {
  return request<SetCameraResponse>('/set-camera', {
    method: 'POST',
    data: body,
  })
}

/** GET /alerts (legacy) */
export function getAlerts(): Promise<AlertsResponse> {
  return request<AlertsResponse>('/alerts')
}

/** GET /recognized (legacy) */
export function getRecognized(): Promise<RecognizedResponse> {
  return request<RecognizedResponse>('/recognized')
}

/** POST /analyze-frame (legacy) */
export function postAnalyzeFrame(formData: FormData): Promise<AnalyzeFrameResponse> {
  return request<AnalyzeFrameResponse>('/analyze-frame', {
    method: 'POST',
    data: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

/** POST /add-known-user (legacy) */
export function postAddKnownUser(body: AddKnownUserRequest): Promise<AddKnownUserResponse> {
  return request<AddKnownUserResponse>('/add-known-user', {
    method: 'POST',
    data: body,
  })
}

/** POST /report-intrusion (legacy) */
export function postReportIntrusion(body: ReportIntrusionRequest): Promise<ReportIntrusionResponse> {
  return request<ReportIntrusionResponse>('/report-intrusion', {
    method: 'POST',
    data: body,
  })
}

/** POST /auth/signup */
export function postSignUp(body: SignUpRequest): Promise<TokenResponse> {
  return request<TokenResponse>('/auth/signup', {
    method: 'POST',
    data: body,
  })
}

/** POST /auth/login */
export function postLogin(body: LoginRequest): Promise<TokenResponse> {
  return request<TokenResponse>('/auth/login', {
    method: 'POST',
    data: body,
  })
}

/** GET /gallery */
export function getGallery(): Promise<GalleryListResponse> {
  return request<GalleryListResponse>('/gallery')
}

/** POST /gallery */
export function postGalleryPerson(formData: FormData): Promise<GalleryPerson> {
  return request<GalleryPerson>('/gallery', {
    method: 'POST',
    data: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

/** PUT /gallery/{id} */
export function putGalleryPerson(id: number, personName: string): Promise<GalleryPerson> {
  return request<GalleryPerson>(`/gallery/${id}`, {
    method: 'PUT',
    data: { person_name: personName },
  })
}

/** DELETE /gallery/{id} */
export function deleteGalleryPerson(id: number): Promise<{ deleted: boolean; message: string }> {
  return request(`/gallery/${id}`, { method: 'DELETE' })
}

/** POST /monitor/frame */
export function postMonitorFrame(formData: FormData): Promise<MonitorFrameResponse> {
  return request<MonitorFrameResponse>('/monitor/frame', {
    method: 'POST',
    data: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

/** GET /recognitions */
export function getRecognitions(): Promise<RecognitionListResponse> {
  return request<RecognitionListResponse>('/recognitions')
}

/** GET /unknowns */
export function getUnknowns(): Promise<UnknownListResponse> {
  return request<UnknownListResponse>('/unknowns')
}

/** GET /security-events */
export function getSecurityEvents(): Promise<SecurityEventListResponse> {
  return request<SecurityEventListResponse>('/security-events')
}

/** POST /unknowns/add-to-gallery */
export function postUnknownAddToGallery(body: AddToGalleryRequest): Promise<{ message: string }> {
  return request('/unknowns/add-to-gallery', {
    method: 'POST',
    data: body,
  })
}

/** POST /unknowns/trigger-alarm */
export function postUnknownTriggerAlarm(body: UnknownActionRequest): Promise<ActionMessageResponse> {
  return request<ActionMessageResponse>('/unknowns/trigger-alarm', {
    method: 'POST',
    data: body,
  })
}

/** POST /unknowns/ignore */
export function postUnknownIgnore(body: UnknownActionRequest): Promise<ActionMessageResponse> {
  return request<ActionMessageResponse>('/unknowns/ignore', {
    method: 'POST',
    data: body,
  })
}

export const api = {
  getHealth,
  postPredict,
  postCompareFaces,
  getVideoFeedStatus,
  getCameras,
  postSetCamera,
  getAlerts,
  getRecognized,
  postAnalyzeFrame,
  postAddKnownUser,
  postReportIntrusion,
  postSignUp,
  postLogin,
  getGallery,
  postGalleryPerson,
  putGalleryPerson,
  deleteGalleryPerson,
  postMonitorFrame,
  getRecognitions,
  getUnknowns,
  getSecurityEvents,
  postUnknownAddToGallery,
  postUnknownTriggerAlarm,
  postUnknownIgnore,
} as const

export type { ApiErrorDetail }
