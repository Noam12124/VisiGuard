import { BellRing, ShieldAlert, UserPlus, UserX } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import type { UnknownDetectionWithMeta } from '@/hooks/use-monitoring'
import { playIntrusionAlarm } from '@/lib/play-alarm'
import {
  getUploadUrl,
  postUnknownAddToGallery,
  postUnknownIgnore,
  postUnknownTriggerAlarm,
} from '@/services/api'

interface UnknownAlertModalProps {
  open: boolean
  payload: UnknownDetectionWithMeta | null
  onClose: () => void
  onActionComplete: () => void
  onAlarmTriggered: () => void
}

export function UnknownAlertModal({
  open,
  payload,
  onClose,
  onActionComplete,
  onAlarmTriggered,
}: UnknownAlertModalProps) {
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)

  if (!open || !payload) return null

  const { detection } = payload
  const unknownId = detection.status === 'unknown' ? detection.unknown_id : 0
  const snapshotUrl =
    detection.status === 'unknown'
      ? getUploadUrl(detection.snapshot_path)
      : undefined

  const handleAddToGallery = async () => {
    if (!name.trim() || !unknownId) return
    setSubmitting(true)
    try {
      const res = await postUnknownAddToGallery({
        unknown_id: unknownId,
        person_name: name.trim(),
      })
      toast.success(res.message)
      onActionComplete()
    } catch {
      toast.error('Failed to add to gallery')
    } finally {
      setSubmitting(false)
    }
  }

  const handleTriggerAlarm = async () => {
    if (!unknownId) return
    setSubmitting(true)
    try {
      await postUnknownTriggerAlarm({ unknown_id: unknownId })
      playIntrusionAlarm()
      onAlarmTriggered()
      toast.error('Security alarm triggered!')
      onActionComplete()
    } catch {
      toast.error('Failed to trigger alarm')
    } finally {
      setSubmitting(false)
    }
  }

  const handleIgnore = async () => {
    if (!unknownId) return
    setSubmitting(true)
    try {
      await postUnknownIgnore({ unknown_id: unknownId })
      toast.message('Event ignored')
      onActionComplete()
    } catch {
      toast.error('Failed to ignore event')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
    >
      <Card className="w-full max-w-lg border-destructive/40 shadow-2xl">
        <CardHeader>
          <div className="mb-2 flex size-10 items-center justify-center rounded-lg bg-destructive/15">
            <ShieldAlert className="size-5 text-destructive" />
          </div>
          <CardTitle>Unknown person detected</CardTitle>
          <CardDescription>
            Confidence: {(detection.confidence * 100).toFixed(1)}% — below recognition threshold
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {snapshotUrl && (
            <img
              src={snapshotUrl}
              alt="Unknown person snapshot"
              className="mx-auto max-h-48 rounded-lg border border-border object-contain"
            />
          )}
          <p className="text-xs text-muted-foreground">
            Detected at {new Date().toLocaleString()}
          </p>
          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="alert-name">
              Name (for Add to Gallery)
            </label>
            <input
              id="alert-name"
              type="text"
              placeholder="e.g. Guest"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-2">
          <Button
            className="w-full"
            disabled={!name.trim() || submitting}
            onClick={handleAddToGallery}
          >
            <UserPlus className="size-4" />
            Add to Gallery
          </Button>
          <Button
            variant="destructive"
            className="w-full"
            disabled={submitting}
            onClick={handleTriggerAlarm}
          >
            <BellRing className="size-4" />
            Trigger Alarm
          </Button>
          <Button
            variant="outline"
            className="w-full"
            disabled={submitting}
            onClick={handleIgnore}
          >
            <UserX className="size-4" />
            Ignore
          </Button>
          <Button variant="ghost" className="w-full" onClick={onClose}>
            Dismiss
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
