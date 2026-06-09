import { useCallback, useEffect, useRef, useState } from 'react'
import { captureFrameFromImage } from '@/lib/capture-frame'
import { postAnalyzeFrame } from '@/services/api'
import type { AnalyzeFrameResponse } from '@/types/api'

const INTERVAL_MS = 2000
const UNKNOWN_COOLDOWN_MS = 15000

export type MonitoringStatus = 'off' | 'active' | 'error' | 'analyzing'

interface UseSurveillanceOptions {
  imageRef: React.RefObject<HTMLImageElement | null>
  enabled: boolean
  onUnknown: (result: AnalyzeFrameResponse) => void
  onAuthorized?: (result: AnalyzeFrameResponse) => void
}

export function useSurveillance({
  imageRef,
  enabled,
  onUnknown,
  onAuthorized,
}: UseSurveillanceOptions) {
  const [monitoringStatus, setMonitoringStatus] =
    useState<MonitoringStatus>('off')
  const [lastResult, setLastResult] = useState<AnalyzeFrameResponse | null>(null)
  const analyzingRef = useRef(false)
  const lastUnknownAtRef = useRef(0)
  const onUnknownRef = useRef(onUnknown)
  const onAuthorizedRef = useRef(onAuthorized)

  useEffect(() => {
    onUnknownRef.current = onUnknown
    onAuthorizedRef.current = onAuthorized
  }, [onUnknown, onAuthorized])

  const tick = useCallback(async () => {
    const img = imageRef.current
    if (!img || analyzingRef.current) return

    const blob = await captureFrameFromImage(img)
    if (!blob) {
      setMonitoringStatus('error')
      return
    }

    analyzingRef.current = true
    setMonitoringStatus('analyzing')

    try {
      const form = new FormData()
      form.append('frame', blob, 'snapshot.jpg')
      const result = await postAnalyzeFrame(form)
      setLastResult(result)
      setMonitoringStatus('active')

      if (result.status === 'authorized') {
        onAuthorizedRef.current?.(result)
      } else {
        const now = Date.now()
        if (now - lastUnknownAtRef.current >= UNKNOWN_COOLDOWN_MS) {
          lastUnknownAtRef.current = now
          onUnknownRef.current(result)
        }
      }
    } catch {
      setMonitoringStatus('error')
    } finally {
      analyzingRef.current = false
    }
  }, [imageRef])

  useEffect(() => {
    if (!enabled) {
      setMonitoringStatus('off')
      return
    }

    setMonitoringStatus('active')
    void tick()
    const id = window.setInterval(() => void tick(), INTERVAL_MS)
    return () => window.clearInterval(id)
  }, [enabled, tick])

  return { monitoringStatus, lastResult }
}
