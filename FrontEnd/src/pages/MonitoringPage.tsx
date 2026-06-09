import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { LiveCameraFeed } from '@/components/camera/LiveCameraFeed'
import { LiveMonitoringIndicator } from '@/components/surveillance/LiveMonitoringIndicator'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { useMonitoringContext } from '@/context/MonitoringContext'
import { useRecognitions } from '@/hooks/use-recognitions'
import { useUnknowns } from '@/hooks/use-unknowns'
import { formatRelativeTime } from '@/lib/format-time'
import { getUploadUrl } from '@/services/api'

export function MonitoringPage() {
  const {
    enabled: surveillanceOn,
    setEnabled: setSurveillanceOn,
    monitoringStatus,
    lastResult,
    alarmActive,
  } = useMonitoringContext()

  const { data: recognitions, isLoading: recLoading } = useRecognitions()
  const { data: unknowns, isLoading: unkLoading } = useUnknowns()

  return (
    <div className={`space-y-6 ${alarmActive ? 'ring-2 ring-destructive rounded-lg p-2' : ''}`}>
      {alarmActive && (
        <div className="rounded-lg bg-destructive/20 p-3 text-center text-destructive">
          SECURITY ALARM ACTIVE
        </div>
      )}

      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Live Monitoring</h1>
        <p className="text-sm text-muted-foreground">
          Surveillance runs globally across all tabs — YOLO detection, embedding match, gallery lookup.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card/50 px-4 py-3">
        <LiveMonitoringIndicator status={monitoringStatus} />
        <label className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
          <input
            type="checkbox"
            checked={surveillanceOn}
            onChange={(e) => setSurveillanceOn(e.target.checked)}
            className="size-4 rounded border-border accent-primary"
          />
          Continuous monitoring (background)
        </label>
        {lastResult && (
          <span className="text-xs text-muted-foreground">
            Last scan: {lastResult.faces_detected} face(s), threshold{' '}
            {(lastResult.threshold * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <LiveCameraFeed compact />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recognition history</CardTitle>
          </CardHeader>
          <CardContent>
            {recLoading && <LoadingSpinner label="Loading…" />}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 pr-4">Person</th>
                    <th className="pb-2 pr-4">Time</th>
                    <th className="pb-2">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {recognitions?.items.map((item) => (
                    <tr key={item.id} className="border-b border-border/50">
                      <td className="py-2 pr-4">{item.person_name}</td>
                      <td className="py-2 pr-4">{formatRelativeTime(item.timestamp)}</td>
                      <td className="py-2">{(item.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {recognitions?.items.length === 0 && (
                <p className="py-4 text-sm text-muted-foreground">No recognitions yet.</p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Unknown persons</CardTitle>
          </CardHeader>
          <CardContent>
            {unkLoading && <LoadingSpinner label="Loading…" />}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 pr-4">Snapshot</th>
                    <th className="pb-2 pr-4">Time</th>
                    <th className="pb-2 pr-4">Action</th>
                    <th className="pb-2">Similarity</th>
                  </tr>
                </thead>
                <tbody>
                  {unknowns?.items.map((item) => (
                    <tr key={item.id} className="border-b border-border/50">
                      <td className="py-2 pr-4">
                        <img
                          src={getUploadUrl(item.snapshot_path)}
                          alt="Unknown"
                          className="size-10 rounded object-cover"
                        />
                      </td>
                      <td className="py-2 pr-4">{formatRelativeTime(item.timestamp)}</td>
                      <td className="py-2 pr-4">{item.action_taken ?? 'pending'}</td>
                      <td className="py-2">{(item.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {unknowns?.items.length === 0 && (
                <p className="py-4 text-sm text-muted-foreground">No unknown detections.</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
