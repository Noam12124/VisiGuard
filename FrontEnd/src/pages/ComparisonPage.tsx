import { Loader2, ScanFace } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { ImageDropzone } from '@/components/comparison/ImageDropzone'
import { VerificationResult } from '@/components/comparison/VerificationResult'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { useCompareFaces } from '@/hooks/use-compare-faces'

export function ComparisonPage() {
  const [fileA, setFileA] = useState<File | null>(null)
  const [fileB, setFileB] = useState<File | null>(null)
  const [previewA, setPreviewA] = useState<string | null>(null)
  const [previewB, setPreviewB] = useState<string | null>(null)

  const compare = useCompareFaces()

  const revokePreview = useCallback((url: string | null) => {
    if (url) URL.revokeObjectURL(url)
  }, [])

  useEffect(() => {
    return () => {
      revokePreview(previewA)
      revokePreview(previewB)
    }
  }, [previewA, previewB, revokePreview])

  const clearA = () => {
    revokePreview(previewA)
    setFileA(null)
    setPreviewA(null)
  }

  const clearB = () => {
    revokePreview(previewB)
    setFileB(null)
    setPreviewB(null)
  }

  const handleCompare = () => {
    if (!fileA || !fileB) return
    const form = new FormData()
    form.append('image_a', fileA)
    form.append('image_b', fileB)
    compare.mutate(form)
  }

  const canSubmit = Boolean(fileA && fileB) && !compare.isPending

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Face Verification</h1>
        <p className="mt-1 text-muted-foreground">
          Upload two face images to compare embeddings via{' '}
          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
            POST /compare-faces
          </code>
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <ImageDropzone
          label="Image A"
          file={fileA}
          previewUrl={previewA}
          onFileSelect={(f, url) => {
            revokePreview(previewA)
            setFileA(f)
            setPreviewA(url)
          }}
          onClear={clearA}
        />
        <ImageDropzone
          label="Image B"
          file={fileB}
          previewUrl={previewB}
          onFileSelect={(f, url) => {
            revokePreview(previewB)
            setFileB(f)
            setPreviewB(url)
          }}
          onClear={clearB}
        />
      </div>

      <div className="flex justify-center">
        <Button size="lg" disabled={!canSubmit} onClick={handleCompare}>
          {compare.isPending ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              Calculating...
            </>
          ) : (
            <>
              <ScanFace className="size-4" />
              Compare Faces
            </>
          )}
        </Button>
      </div>

      {compare.isPending && (
        <Card>
          <CardHeader className="text-center">
            <CardTitle>Calculating...</CardTitle>
            <CardDescription>
              Extracting embeddings and computing cosine similarity
            </CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center py-8">
            <Loader2 className="size-10 animate-spin text-primary" />
          </CardContent>
        </Card>
      )}

      {compare.isSuccess && compare.data && (
        <VerificationResult data={compare.data} />
      )}
    </div>
  )
}
