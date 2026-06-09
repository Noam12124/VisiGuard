import { Camera, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useCameras, useSetCamera } from '@/hooks/use-cameras'
import { cn } from '@/lib/utils'

interface CameraSourcePickerProps {
  onSourceChanged?: () => void
}

export function CameraSourcePicker({ onSourceChanged }: CameraSourcePickerProps) {
  const { data, isLoading } = useCameras()
  const setCamera = useSetCamera()

  const activeIndex = data?.active_index ?? 0

  const handleSelect = (index: number, available: boolean) => {
    if (!available || setCamera.isPending) return
    setCamera.mutate(
      { source_index: index },
      { onSuccess: () => onSourceChanged?.() },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Camera className="size-4 text-primary" />
          Video source
        </CardTitle>
        <CardDescription>
          Switch between webcam indices (e.g. default vs Bluetooth bridge). The
          previous device is released before opening the new one.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          {data?.sources.map((source) => {
            const isActive = source.index === activeIndex
            return (
              <Button
                key={source.index}
                type="button"
                variant={isActive ? 'default' : 'outline'}
                size="sm"
                disabled={!source.available || setCamera.isPending}
                className={cn(
                  'gap-2',
                  !source.available && 'opacity-50',
                )}
                onClick={() => handleSelect(source.index, source.available)}
              >
                {setCamera.isPending && isActive && (
                  <Loader2 className="size-3 animate-spin" />
                )}
                {source.label}
              </Button>
            )
          })}
        </div>

        {data && (
          <p className="mt-3 text-xs text-muted-foreground">
            Active source: index {data.active_index}
            {setCamera.isPending && ' · Switching...'}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
