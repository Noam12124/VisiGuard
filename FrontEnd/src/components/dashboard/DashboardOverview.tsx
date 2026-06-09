import {
  Eye,
  ScanFace,
  ShieldCheck,
  ShieldOff,
  Users,
} from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useAlerts } from '@/hooks/use-alerts'
import { useRecognized } from '@/hooks/use-recognized'
import { useVideoFeedStatus } from '@/hooks/use-video-feed-status'

export function DashboardOverview() {
  const { data: alerts, isLoading: alertsLoading } = useAlerts()
  const { data: recognized, isLoading: recognizedLoading } = useRecognized()
  const { data: cameraStatus, isLoading: cameraLoading } = useVideoFeedStatus()

  const alertCount = alerts?.items.length ?? 0
  const recognitionCount = recognized?.items.length ?? 0
  const camerasActive = cameraStatus?.connected ? '1' : '0'

  const stats = [
    {
      label: 'Cameras Active',
      value: cameraLoading ? null : camerasActive,
      description: cameraStatus?.connected
        ? 'Live stream connected'
        : 'Camera offline or unavailable',
      icon: Eye,
    },
    {
      label: 'Recognitions Logged',
      value: recognizedLoading ? null : String(recognitionCount),
      description: 'Authorized matches in event log',
      icon: ScanFace,
    },
    {
      label: 'Active Alerts',
      value: alertsLoading ? null : String(alertCount),
      description:
        alertCount > 0
          ? 'Review unknown-person events'
          : 'No unknown persons detected',
      icon: ShieldOff,
    },
    {
      label: 'Event Sources',
      value: '1',
      description: `Camera index ${cameraStatus?.source_index ?? 0}`,
      icon: Users,
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Dashboard</h2>
        <p className="mt-1 text-muted-foreground">
          Real-time overview of your VisiGuard security system.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map(({ label, value, description, icon: Icon }) => (
          <Card key={label}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardDescription>{label}</CardDescription>
                <Icon className="size-4 text-muted-foreground" />
              </div>
              <CardTitle className="text-3xl font-bold tabular-nums">
                {value === null ? (
                  <Skeleton className="h-9 w-12" />
                ) : (
                  value
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">{description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-primary/20 bg-primary/5">
        <CardContent className="flex items-start gap-4 p-6">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/15">
            <ShieldCheck className="size-5 text-primary" />
          </div>
          <div>
            <p className="font-medium">System armed & monitoring</p>
            <p className="mt-1 text-sm text-muted-foreground">
              VisiGuard combines camera feeds with AI-powered face recognition
              to distinguish authorized family members from unknown visitors,
              alerting you in real time when someone unrecognized is detected.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
