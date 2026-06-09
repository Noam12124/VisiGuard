import { ShieldAlert, UserPlus, UserX } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import type { AnalyzeFrameResponse } from '@/types/api'

interface UnknownPersonModalProps {
  open: boolean
  result: AnalyzeFrameResponse | null
  onAuthorize: (name: string) => void
  onReject: () => void
  onClose: () => void
  isSubmitting?: boolean
}

export function UnknownPersonModal({
  open,
  result,
  onAuthorize,
  onReject,
  onClose,
  isSubmitting = false,
}: UnknownPersonModalProps) {
  const [name, setName] = useState('')

  if (!open || !result) return null

  const scoreText =
    result.similarity_score != null
      ? `${(result.similarity_score * 100).toFixed(1)}% (threshold ${(result.threshold * 100).toFixed(0)}%)`
      : 'No gallery match'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="unknown-person-title"
    >
      <Card className="w-full max-w-md border-destructive/40 shadow-2xl">
        <CardHeader>
          <div className="mb-2 flex size-10 items-center justify-center rounded-lg bg-destructive/15">
            <ShieldAlert className="size-5 text-destructive" />
          </div>
          <CardTitle id="unknown-person-title">
            Unknown person detected
          </CardTitle>
          <CardDescription>
            Authorize access? Similarity: {scoreText}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3">
          <label className="text-sm font-medium" htmlFor="authorize-name">
            Name (if authorizing)
          </label>
          <input
            id="authorize-name"
            type="text"
            placeholder="e.g. Family Member"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </CardContent>

        <CardFooter className="flex flex-col gap-2 sm:flex-row">
          <Button
            className="w-full sm:flex-1"
            disabled={!name.trim() || isSubmitting || !result.embedding?.length}
            onClick={() => onAuthorize(name.trim())}
          >
            <UserPlus className="size-4" />
            Authorize
          </Button>
          <Button
            variant="destructive"
            className="w-full sm:flex-1"
            disabled={isSubmitting}
            onClick={onReject}
          >
            <UserX className="size-4" />
            Block / Reject
          </Button>
          <Button variant="ghost" className="w-full sm:w-auto" onClick={onClose}>
            Dismiss
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
