import axios, { type AxiosError, type AxiosRequestConfig } from 'axios'
import { clearAuth, getAccessToken } from '@/lib/auth-storage'
import type { ApiErrorDetail } from '@/types/api'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
  readonly status: number
  readonly body?: ApiErrorDetail

  constructor(message: string, status: number, body?: ApiErrorDetail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { Accept: 'application/json' },
})

apiClient.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorDetail>) => {
    const path = error.config?.url ?? ''
    if (error.response?.status === 401 && !path.includes('/auth/')) {
      clearAuth()
    }
    const status = error.response?.status ?? 0
    const body = error.response?.data
    const message =
      typeof body?.detail === 'string'
        ? body.detail
        : error.message || `Request failed (${status})`
    return Promise.reject(new ApiError(message, status, body))
  },
)

export async function request<T>(
  path: string,
  config?: AxiosRequestConfig,
): Promise<T> {
  const response = await apiClient.request<T>({ url: path, ...config })
  return response.data
}

export function getApiBaseUrl(): string {
  return API_BASE_URL
}

export function getUploadUrl(relativePath: string): string {
  const normalized = relativePath.replace(/^\/+/, '')
  return `${API_BASE_URL}/uploads/${normalized}`
}

export function getVideoFeedUrl(): string {
  return `${API_BASE_URL}/video-feed`
}
