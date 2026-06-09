import { CheckCircle2, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { cn } from '@/lib/utils'
import type { CompareFacesResponse } from '@/types/api'

interface VerificationResultProps {
  data: CompareFacesResponse
}

export function VerificationResult({ data }: VerificationResultProps) {
  const isMatch = data.is_match

  return (
    <Card
      className={cn(
        'border-2',
        isMatch ? 'border-success/40 bg-success/5' : 'border-destructive/40 bg-destructive/5',
      )}
    >
      <CardHeader className="text-center">
        <div className="mx-auto mb-2 flex size-14 items-center justify-center rounded-full bg-background">
          {isMatch ? (
            <CheckCircle2 className="size-8 text-success" />
          ) : (
            <XCircle className="size-8 text-destructive" />
          )}
        </div>
        <CardTitle className="text-2xl">{data.identity_label}</CardTitle>
        <CardDescription>
          Cosine similarity (scikit-learn) vs threshold {data.threshold.toFixed(3)}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="text-center">
          <p className="text-5xl font-bold tabular-nums tracking-tight">
            {data.match_percent.toFixed(1)}%
          </p>
          <p className="mt-1 text-sm text-muted-foreground">Match score</p>
        </div>

        <div className="h-3 overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              'h-full rounded-full transition-all',
              isMatch ? 'bg-success' : 'bg-destructive',
            )}
            style={{ width: `${Math.min(100, Math.max(0, data.match_percent))}%` }}
          />
        </div>

        <dl className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg bg-muted/50 p-3">
            <dt className="text-xs text-muted-foreground">Similarity score</dt>
            <dd className="mt-1 font-mono text-sm">
              {data.similarity_score.toFixed(4)}
            </dd>
          </div>
          <div className="rounded-lg bg-muted/50 p-3">
            <dt className="text-xs text-muted-foreground">is_match</dt>
            <dd className="mt-1">
              <Badge variant={isMatch ? 'success' : 'destructive'}>
                {String(data.is_match)}
              </Badge>
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  )
}
