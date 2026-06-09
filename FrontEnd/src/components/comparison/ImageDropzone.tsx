import { ImagePlus, Upload, X } from 'lucide-react'
import { useCallback, useId, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { cn } from '@/lib/utils'

const MAX_BYTES = 5 * 1024 * 1024
const ACCEPT = ['image/jpeg', 'image/png', 'image/webp']

interface ImageDropzoneProps {
  label: string
  file: File | null
  previewUrl: string | null
  onFileSelect: (file: File, previewUrl: string) => void
  onClear: () => void
}

export function ImageDropzone({
  label,
  file,
  previewUrl,
  onFileSelect,
  onClear,
}: ImageDropzoneProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const validateAndSet = useCallback(
    (candidate: File | undefined) => {
      if (!candidate) return
      if (!ACCEPT.includes(candidate.type)) {
        toast.error('Invalid file type', {
          description: 'Use JPEG, PNG, or WebP images.',
        })
        return
      }
      if (candidate.size > MAX_BYTES) {
        toast.error('File too large', { description: 'Maximum size is 5 MB.' })
        return
      }
      onFileSelect(candidate, URL.createObjectURL(candidate))
    },
    [onFileSelect],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      validateAndSet(e.dataTransfer.files[0])
    },
    [validateAndSet],
  )

  return (
    <Card
      className={cn(
        'transition-colors',
        dragOver && 'border-primary ring-1 ring-primary/30',
      )}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <CardHeader>
        <CardTitle className="text-base">{label}</CardTitle>
        <CardDescription>Drag & drop or click to upload</CardDescription>
      </CardHeader>
      <CardContent>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPT.join(',')}
          className="sr-only"
          onChange={(e) => validateAndSet(e.target.files?.[0])}
        />

        {previewUrl ? (
          <div className="relative">
            <img
              src={previewUrl}
              alt={`${label} preview`}
              className="aspect-square w-full rounded-lg border border-border object-cover"
            />
            <Button
              type="button"
              variant="secondary"
              size="icon"
              className="absolute right-2 top-2 size-8"
              onClick={onClear}
              aria-label={`Remove ${label}`}
            >
              <X className="size-4" />
            </Button>
            {file && (
              <p className="mt-2 truncate text-xs text-muted-foreground">
                {file.name}
              </p>
            )}
          </div>
        ) : (
          <label
            htmlFor={inputId}
            className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-6 py-12 transition-colors hover:bg-muted/40"
          >
            <div className="flex size-12 items-center justify-center rounded-full bg-primary/10">
              <ImagePlus className="size-6 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium">Drop image here</p>
              <p className="mt-1 text-xs text-muted-foreground">
                or click to browse
              </p>
            </div>
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <Upload className="size-3" />
              Max 5 MB
            </span>
          </label>
        )}
      </CardContent>
    </Card>
  )
}
