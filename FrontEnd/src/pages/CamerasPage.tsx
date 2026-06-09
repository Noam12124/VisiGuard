import { useState } from 'react'
import { CameraSourcePicker } from '@/components/camera/CameraSourcePicker'
import { LiveCameraFeed } from '@/components/camera/LiveCameraFeed'

export function CamerasPage() {
  const [streamKey, setStreamKey] = useState(0)

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Live Cameras</h1>
        <p className="mt-1 text-muted-foreground">
          Real-time MJPEG feed with switchable video sources.
        </p>
      </header>

      <CameraSourcePicker onSourceChanged={() => setStreamKey((k) => k + 1)} />

      <LiveCameraFeed streamKey={streamKey} />
    </div>
  )
}
