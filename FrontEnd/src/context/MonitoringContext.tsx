import { useQueryClient } from '@tanstack/react-query'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { toast } from 'sonner'
import { UnknownAlertModal } from '@/components/surveillance/UnknownAlertModal'
import {
  useMonitoring,
  type MonitoringStatus,
  type UnknownDetectionWithMeta,
} from '@/hooks/use-monitoring'
import { playIntrusionAlarm } from '@/lib/play-alarm'
import { queryKeys } from '@/services/query-keys'
import type { MonitorFrameResponse } from '@/types/auth'

const STORAGE_KEY = 'visiguard.monitoring.enabled'
const INVALIDATE_COOLDOWN_MS = 12000

interface MonitoringContextValue {
  enabled: boolean
  setEnabled: (on: boolean) => void
  monitoringStatus: MonitoringStatus
  lastResult: MonitorFrameResponse | null
  alarmActive: boolean
  clearAlarm: () => void
}

const MonitoringContext = createContext<MonitoringContextValue | null>(null)

function readEnabledDefault(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw === 'false') return false
    if (raw === 'true') return true
  } catch {
    /* ignore */
  }
  return true
}

export function MonitoringProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const [enabled, setEnabledState] = useState(readEnabledDefault)
  const [modalOpen, setModalOpen] = useState(false)
  const [alarmActive, setAlarmActive] = useState(false)
  const [pendingUnknown, setPendingUnknown] =
    useState<UnknownDetectionWithMeta | null>(null)
  const [audioUnlocked, setAudioUnlocked] = useState(false)
  const lastInvalidateRef = useRef(0)

  useEffect(() => {
    const unlock = () => {
      setAudioUnlocked(true)
      window.removeEventListener('click', unlock)
    }
    window.addEventListener('click', unlock)
    return () => window.removeEventListener('click', unlock)
  }, [])

  const setEnabled = useCallback((on: boolean) => {
    setEnabledState(on)
    try {
      localStorage.setItem(STORAGE_KEY, String(on))
    } catch {
      /* ignore */
    }
  }, [])

  const invalidateEvents = useCallback(() => {
    const now = Date.now()
    if (now - lastInvalidateRef.current < INVALIDATE_COOLDOWN_MS) return
    lastInvalidateRef.current = now
    void queryClient.invalidateQueries({ queryKey: queryKeys.recognitions })
    void queryClient.invalidateQueries({ queryKey: queryKeys.unknowns })
  }, [queryClient])

  const handleUnknown = useCallback(
    (payload: UnknownDetectionWithMeta) => {
      setPendingUnknown(payload)
      setModalOpen(true)
      if (audioUnlocked) {
        playIntrusionAlarm()
      } else {
        toast.warning('Unknown person detected! Click anywhere to enable alarm sound.')
      }
      invalidateEvents()
    },
    [audioUnlocked, invalidateEvents],
  )

  const { monitoringStatus, lastResult } = useMonitoring({
    enabled,
    captureSource: 'api',
    onUnknown: handleUnknown,
    onKnown: invalidateEvents,
  })

  const invalidateAll = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.recognitions })
    void queryClient.invalidateQueries({ queryKey: queryKeys.unknowns })
    void queryClient.invalidateQueries({ queryKey: queryKeys.securityEvents })
    void queryClient.invalidateQueries({ queryKey: queryKeys.gallery })
  }, [queryClient])

  const value = useMemo(
    () => ({
      enabled,
      setEnabled,
      monitoringStatus,
      lastResult,
      alarmActive,
      clearAlarm: () => setAlarmActive(false),
    }),
    [enabled, setEnabled, monitoringStatus, lastResult, alarmActive],
  )

  return (
    <MonitoringContext.Provider value={value}>
      {children}
      <UnknownAlertModal
        open={modalOpen}
        payload={pendingUnknown}
        onClose={() => {
          setModalOpen(false)
          setPendingUnknown(null)
        }}
        onActionComplete={() => {
          setModalOpen(false)
          setPendingUnknown(null)
          invalidateAll()
        }}
        onAlarmTriggered={() => setAlarmActive(true)}
      />
    </MonitoringContext.Provider>
  )
}

export function useMonitoringContext(): MonitoringContextValue {
  const ctx = useContext(MonitoringContext)
  if (!ctx) {
    throw new Error('useMonitoringContext must be used within MonitoringProvider')
  }
  return ctx
}
