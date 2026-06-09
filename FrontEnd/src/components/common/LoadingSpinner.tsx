import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LoadingSpinnerProps {
  className?: string
  label?: string
}

export function LoadingSpinner({
  className,
  label = 'Loading',
}: LoadingSpinnerProps) {
  return (
    <div
      className={cn('flex items-center justify-center gap-2', className)}
      role="status"
      aria-label={label}
    >
      <Loader2 className="size-5 animate-spin text-primary" />
      <span className="text-sm text-muted-foreground">{label}</span>
    </div>
  )
}
