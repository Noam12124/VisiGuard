import { AlertCircle, RefreshCw, Video } from 'lucide-react'
import { forwardRef, useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/common/StatusBadge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { useVideoFeedStatus } from '@/hooks/use-video-feed-status'
import { cn } from '@/lib/utils'
import { getVideoFeedUrl } from '@/services/api'

interface LiveCameraFeedProps {
  compact?: boolean
  showHeader?: boolean
  streamKey?: number
  headerExtra?: React.ReactNode
}

export const LiveCameraFeed = forwardRef<HTMLImageElement, LiveCameraFeedProps>(
  function LiveCameraFeed(
    { compact = false, showHeader = true, streamKey: externalStreamKey, headerExtra },
    ref,
  ) {
    const [internalStreamKey, setInternalStreamKey] = useState(0)
    const streamKey = externalStreamKey ?? internalStreamKey
    const [streamError, setStreamError] = useState(false)
    const { data: status, isLoading: statusLoading } = useVideoFeedStatus()

    const streamUrl = `${getVideoFeedUrl()}?t=${streamKey}`

    const handleRetry = useCallback(() => {
      setStreamError(false)
      setInternalStreamKey((k) => k + 1)
    }, [])

    const connected = status?.connected && !streamError
    const feedStatus = statusLoading
      ? 'pending'
      : connected
        ? 'online'
        : 'offline'

    return (
      <Card className={cn('overflow-hidden', compact && 'h-full')}>
        {showHeader && (
          <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0 pb-4">
            <div className="min-w-0 flex-1">
              <CardTitle className="flex items-center gap-2">
                <Video className="size-4 text-primary" />
                Live Camera
              </CardTitle>
              <CardDescription>
                MJPEG preview · analysis runs in background on all tabs
              </CardDescription>
              {headerExtra && <div className="mt-3">{headerExtra}</div>}
            </div>
            <StatusBadge status={feedStatus} pulse={feedStatus === 'online'} />
          </CardHeader>
        )}

        <CardContent className={cn(!showHeader && 'pt-6')}>
          <div
            className={cn(
              'relative overflow-hidden rounded-lg border border-border bg-black',
              compact ? 'aspect-video' : 'aspect-video max-h-[480px]',
            )}
          >
            {!streamError ? (
              <img
                ref={ref}
                key={streamKey}
                src={streamUrl}
                alt="Live security camera feed"
                crossOrigin="anonymous"
                className="size-full object-cover"
                onError={() => setStreamError(true)}
                onLoad={() => setStreamError(false)}
              />
            ) : (
              <div className="flex size-full flex-col items-center justify-center gap-3 p-6 text-center">
                <AlertCircle className="size-10 text-destructive" />
                <p className="text-sm font-medium">Camera stream unavailable</p>
                <p className="max-w-xs text-xs text-muted-foreground">
                  Check that a webcam is connected and the backend is running.
                </p>
                <Button variant="outline" size="sm" onClick={handleRetry}>
                  <RefreshCw className="size-4" />
                  Reconnect
                </Button>
              </div>
            )}

            {statusLoading && !streamError && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/60">
                <LoadingSpinner label="Connecting..." />
              </div>
            )}
          </div>

          {compact && (
            <Button variant="link" className="mt-3 h-auto p-0" asChild>
              <Link to="/cameras">Open full camera view →</Link>
            </Button>
          )}

          {!compact && (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs text-muted-foreground">
                Source index: {status?.source_index ?? '—'}
              </p>
              <Button variant="outline" size="sm" onClick={handleRetry}>
                <RefreshCw className="size-4" />
                Refresh stream
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    )
  },
)
