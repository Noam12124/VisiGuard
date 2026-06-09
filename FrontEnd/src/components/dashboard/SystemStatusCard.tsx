import {
  CheckCircle2,
  RefreshCw,
  Server,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useHealthCheck } from '@/hooks/use-health-check'
import { cn } from '@/lib/utils'

export function SystemStatusCard() {
  const { data, isLoading, isError, error, refetch, isFetching } =
    useHealthCheck()

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2">
              <Server className="size-4 text-primary" />
              API Status
            </CardTitle>
            <CardDescription>
              Live health check against{' '}
              <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                GET /
              </code>
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={() => refetch()}
            disabled={isFetching}
            aria-label="Refresh API status"
          >
            <RefreshCw
              className={cn('size-4', isFetching && 'animate-spin')}
            />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading && (
          <div className="space-y-3">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        )}

        {isError && (
          <div className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4">
            <WifiOff className="mt-0.5 size-5 shrink-0 text-destructive" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-destructive">
                Backend unreachable
              </p>
              <p className="text-sm text-muted-foreground">
                {error instanceof Error
                  ? error.message
                  : 'Could not connect to the API at http://127.0.0.1:8000'}
              </p>
              <p className="text-xs text-muted-foreground">
                Make sure FastAPI is running:{' '}
                <code className="rounded bg-muted px-1 py-0.5">
                  uvicorn main:app --reload
                </code>
              </p>
            </div>
          </div>
        )}

        {data && (
          <div className="space-y-4">
            <div className="flex items-start gap-3 rounded-lg border border-success/20 bg-success/5 p-4 glow-emerald">
              <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-success" />
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Wifi className="size-3.5 text-success" />
                  <p className="text-sm font-medium">Connected</p>
                </div>
                <p className="text-sm text-muted-foreground">{data.message}</p>
              </div>
            </div>

            <dl className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg bg-muted/50 p-3">
                <dt className="text-xs text-muted-foreground">Endpoint</dt>
                <dd className="mt-1 font-mono text-sm">http://127.0.0.1:8000</dd>
              </div>
              <div className="rounded-lg bg-muted/50 p-3">
                <dt className="text-xs text-muted-foreground">Auto-refresh</dt>
                <dd className="mt-1 text-sm">Every 30 seconds</dd>
              </div>
            </dl>
          </div>
        )}

        {isFetching && !isLoading && (
          <LoadingSpinner className="py-2" label="Refreshing..." />
        )}
      </CardContent>
    </Card>
  )
}
