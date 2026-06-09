import { BrainCircuit, Loader2, Sparkles, Zap } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { usePredict } from '@/hooks/use-predict'

export function PredictPanel() {
  const { mutate, isPending, data, isSuccess } = usePredict()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BrainCircuit className="size-4 text-primary" />
          Model Inference
        </CardTitle>
        <CardDescription>
          Trigger the embedding model via{' '}
          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
            POST /predict
          </code>
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="rounded-lg border border-border bg-muted/30 p-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Sparkles className="size-4 text-primary" />
            Pipeline stages
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {['Person Detection', 'Feature Extraction', 'Face Recognition', 'Decision'].map(
              (stage) => (
                <Badge key={stage} variant="secondary">
                  {stage}
                </Badge>
              ),
            )}
          </div>
        </div>

        <Separator />

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm text-muted-foreground">
            Send a prediction request to verify the model endpoint is wired up.
          </p>
          <Button
            onClick={() => mutate()}
            disabled={isPending}
            className="shrink-0"
          >
            {isPending ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Zap className="size-4" />
                Run Prediction
              </>
            )}
          </Button>
        </div>

        {isSuccess && data && (
          <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Last response
            </p>
            <p className="mt-1 font-mono text-sm text-primary">{data.status}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
