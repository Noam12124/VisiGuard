import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { useSecurityEvents } from '@/hooks/use-security-events'
import { useUnknowns } from '@/hooks/use-unknowns'
import { formatRelativeTime } from '@/lib/format-time'

export function SecurityAlertsPage() {
  const { data: securityEvents, isLoading: secLoading } = useSecurityEvents()
  const { data: unknowns, isLoading: unkLoading } = useUnknowns()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Security Alerts</h1>
        <p className="text-sm text-muted-foreground">
          Alarm events and unknown person resolutions.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Security events</CardTitle>
        </CardHeader>
        <CardContent>
          {secLoading && <LoadingSpinner label="Loading events…" />}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="pb-2 pr-4">Alert time</th>
                  <th className="pb-2 pr-4">Event type</th>
                  <th className="pb-2">Resolution</th>
                </tr>
              </thead>
              <tbody>
                {securityEvents?.items.map((event) => (
                  <tr key={event.id} className="border-b border-border/50">
                    <td className="py-2 pr-4">{formatRelativeTime(event.timestamp)}</td>
                    <td className="py-2 pr-4 capitalize">{event.event_type}</td>
                    <td className="py-2">{event.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {securityEvents?.items.length === 0 && (
              <p className="py-4 text-sm text-muted-foreground">No security events logged.</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Unknown person actions</CardTitle>
        </CardHeader>
        <CardContent>
          {unkLoading && <LoadingSpinner label="Loading…" />}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="pb-2 pr-4">Time</th>
                  <th className="pb-2 pr-4">Confidence</th>
                  <th className="pb-2">Action taken</th>
                </tr>
              </thead>
              <tbody>
                {unknowns?.items
                  .filter((u) => u.action_taken)
                  .map((item) => (
                    <tr key={item.id} className="border-b border-border/50">
                      <td className="py-2 pr-4">{formatRelativeTime(item.timestamp)}</td>
                      <td className="py-2 pr-4">{(item.confidence * 100).toFixed(1)}%</td>
                      <td className="py-2 capitalize">{item.action_taken?.replace(/_/g, ' ')}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
