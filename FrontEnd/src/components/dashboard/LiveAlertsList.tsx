import { AlertTriangle, Bell } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useAlerts } from '@/hooks/use-alerts'
import { formatRelativeTime } from '@/lib/format-time'
import type { AlertSeverity } from '@/types/api'

const severityVariant: Record<
  AlertSeverity,
  'destructive' | 'warning' | 'secondary'
> = {
  high: 'destructive',
  medium: 'warning',
  low: 'secondary',
}

export function LiveAlertsList() {
  const { data, isLoading, isError } = useAlerts()
  const items = data?.items ?? []

  return (
    <Card className="flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="size-4 text-destructive" />
          Live Alerts
        </CardTitle>
        <CardDescription>
          Unknown persons and security events — refreshes every 3s
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1">
        {isLoading && (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        )}

        {isError && (
          <p className="text-sm text-destructive">Failed to load alerts.</p>
        )}

        {!isLoading && !isError && items.length === 0 && (
          <div className="flex flex-col items-center py-8 text-center">
            <div className="flex size-10 items-center justify-center rounded-full bg-muted">
              <Bell className="size-5 text-muted-foreground" />
            </div>
            <p className="mt-3 text-sm font-medium">No active threats</p>
            <p className="text-xs text-muted-foreground">
              Alerts appear when unknown persons are detected.
            </p>
          </div>
        )}

        <ul className="max-h-80 space-y-2 overflow-y-auto">
          {items.map((alert) => (
            <li
              key={alert.id}
              className="flex items-start gap-3 rounded-lg border border-border bg-muted/20 p-3"
            >
              <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">
                  {alert.person_label}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatRelativeTime(alert.timestamp)} · {alert.camera_id}
                  {alert.confidence != null &&
                    ` · ${(alert.confidence * 100).toFixed(0)}% conf.`}
                </p>
              </div>
              <Badge variant={severityVariant[alert.severity]} className="shrink-0">
                {alert.severity}
              </Badge>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}
