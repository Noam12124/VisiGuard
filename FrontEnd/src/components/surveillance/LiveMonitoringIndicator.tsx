import { cn } from '@/lib/utils'
import type { MonitoringStatus } from '@/hooks/use-monitoring'

interface LiveMonitoringIndicatorProps {
  status: MonitoringStatus
  compact?: boolean
}

const labels: Record<MonitoringStatus, string> = {
  off: 'Monitoring off',
  active: 'Live monitoring',
  error: 'Monitoring error',
}

export function LiveMonitoringIndicator({
  status,
  compact = false,
}: LiveMonitoringIndicatorProps) {
  const isLive = status === 'active'
  const isError = status === 'error'

  return (
    <div
      className={cn(
        'inline-flex items-center gap-2 rounded-full border text-xs font-medium',
        compact ? 'px-2 py-1' : 'px-3 py-1.5',
        isLive && 'border-success/30 bg-success/10 text-success',
        isError && 'border-destructive/30 bg-destructive/10 text-destructive',
        status === 'off' && 'border-border bg-muted/50 text-muted-foreground',
      )}
      title={labels[status]}
    >
      <span
        className={cn(
          'size-2 shrink-0 rounded-full',
          isLive && 'bg-success',
          isError && 'bg-destructive',
          status === 'off' && 'bg-muted-foreground',
        )}
      />
      {!compact && labels[status]}
    </div>
  )
}
