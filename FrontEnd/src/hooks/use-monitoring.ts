import { useCallback, useEffect, useRef, useState } from 'react'
import { captureFrameFromApi, captureFrameFromImage } from '@/lib/capture-frame'
import { postMonitorFrame } from '@/services/api'
import type { FaceDetection, MonitorFrameResponse } from '@/types/auth'

/** Minimum pause between scans (ms). */
const MIN_GAP_MS = 2500
/** Target cadence when inference is fast (ms). */
const TARGET_CADENCE_MS = 5000
const UNKNOWN_COOLDOWN_MS = 15000

export type MonitoringStatus = 'off' | 'active' | 'error'

export type CaptureSource = 'api' | 'image'

export interface UnknownDetectionWithMeta {
  detection: FaceDetection & { status: 'unknown' }
  result: MonitorFrameResponse
}

interface UseMonitoringOptions {
  enabled: boolean
  captureSource?: CaptureSource
  imageRef?: React.RefObject<HTMLImageElement | null>
  onUnknown: (detection: UnknownDetectionWithMeta) => void
  onKnown?: (detection: FaceDetection) => void
}

async function captureFrame(
  source: CaptureSource,
  imageRef?: React.RefObject<HTMLImageElement | null>,
): Promise<Blob | null> {
  if (source === 'image' && imageRef?.current) {
    const blob = await captureFrameFromImage(imageRef.current, 0.7)
    if (blob) return blob
  }
  return captureFrameFromApi()
}

/**
 * Background monitoring loop — one scan at a time, no overlapping requests,
 * minimal React state updates to avoid UI jank.
 */
export function useMonitoring({
  enabled,
  captureSource = 'api',
  imageRef,
  onUnknown,
  onKnown,
}: UseMonitoringOptions) {
  const [monitoringStatus, setMonitoringStatus] =
    useState<MonitoringStatus>('off')
  const [lastResult, setLastResult] = useState<MonitorFrameResponse | null>(null)
  const lastUnknownAtRef = useRef(0)
  const onUnknownRef = useRef(onUnknown)
  const onKnownRef = useRef(onKnown)
  const consecutiveErrorsRef = useRef(0)

  useEffect(() => {
    onUnknownRef.current = onUnknown
    onKnownRef.current = onKnown
  }, [onUnknown, onKnown])

  const tick = useCallback(async () => {
    const blob = await captureFrame(captureSource, imageRef)
    if (!blob) {
      consecutiveErrorsRef.current += 1
      if (consecutiveErrorsRef.current >= 3) {
        setMonitoringStatus('error')
      }
      return
    }

    try {
      const form = new FormData()
      form.append('frame', blob, 'snapshot.jpg')
      const result = await postMonitorFrame(form)

      consecutiveErrorsRef.current = 0
      setMonitoringStatus('active')
      setLastResult(result)

      for (const detection of result.detections) {
        if (detection.status === 'known') {
          onKnownRef.current?.(detection)
        } else if (detection.status === 'unknown') {
          const now = Date.now()
          if (now - lastUnknownAtRef.current >= UNKNOWN_COOLDOWN_MS) {
            lastUnknownAtRef.current = now
            onUnknownRef.current({ detection, result })
          }
        }
      }
    } catch {
      consecutiveErrorsRef.current += 1
      if (consecutiveErrorsRef.current >= 3) {
        setMonitoringStatus('error')
      }
    }
  }, [captureSource, imageRef])

  useEffect(() => {
    if (!enabled) {
      setMonitoringStatus('off')
      return
    }

    setMonitoringStatus('active')
    let cancelled = false
    let timerId = 0

    const loop = async () => {
      while (!cancelled) {
        const started = Date.now()
        await tick()
        if (cancelled) break
        const elapsed = Date.now() - started
        const wait = Math.max(MIN_GAP_MS, TARGET_CADENCE_MS - elapsed)
        await new Promise<void>((resolve) => {
          timerId = window.setTimeout(resolve, wait)
        })
      }
    }

    void loop()
    return () => {
      cancelled = true
      window.clearTimeout(timerId)
    }
  }, [enabled, tick])

  return { monitoringStatus, lastResult }
}
