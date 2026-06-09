import { LiveAlertsList } from '@/components/dashboard/LiveAlertsList'
import { RecognizedPeopleList } from '@/components/dashboard/RecognizedPeopleList'

export function AlertsPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Security Events</h1>
        <p className="mt-1 text-muted-foreground">
          Live alert feed and recently identified authorized persons.
        </p>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <LiveAlertsList />
        <RecognizedPeopleList />
      </div>
    </div>
  )
}
