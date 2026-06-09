import { ScanFace, UserCheck } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useRecognized } from '@/hooks/use-recognized'
import { formatRelativeTime } from '@/lib/format-time'

export function RecognizedPeopleList() {
  const { data, isLoading, isError } = useRecognized()
  const items = data?.items ?? []

  return (
    <Card className="flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <UserCheck className="size-4 text-success" />
          Recently Identified
        </CardTitle>
        <CardDescription>
          Authorized recognitions from the pipeline — refreshes every 3s
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1">
        {isLoading && (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        )}

        {isError && (
          <p className="text-sm text-destructive">Failed to load recognitions.</p>
        )}

        {!isLoading && !isError && items.length === 0 && (
          <div className="flex flex-col items-center py-8 text-center">
            <ScanFace className="size-10 text-muted-foreground" />
            <p className="mt-3 text-sm font-medium">No recognitions yet</p>
            <p className="text-xs text-muted-foreground">
              Authorized matches will appear here in real time.
            </p>
          </div>
        )}

        <ul className="max-h-80 space-y-2 overflow-y-auto">
          {items.map((person) => (
            <li
              key={person.id}
              className="flex items-center gap-3 rounded-lg border border-border bg-muted/20 p-3"
            >
              <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-success/15">
                <UserCheck className="size-4 text-success" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">
                  {person.person_name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatRelativeTime(person.timestamp)} · {person.camera_id}
                </p>
              </div>
              <Badge variant="success" className="shrink-0 tabular-nums">
                {(person.similarity_score * 100).toFixed(0)}%
              </Badge>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}
