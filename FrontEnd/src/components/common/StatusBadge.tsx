import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

type StatusVariant = 'online' | 'offline' | 'pending' | 'warning'

const variantMap: Record<
  StatusVariant,
  { label: string; variant: 'success' | 'destructive' | 'secondary' | 'warning' }
> = {
  online: { label: 'Online', variant: 'success' },
  offline: { label: 'Offline', variant: 'destructive' },
  pending: { label: 'Checking', variant: 'secondary' },
  warning: { label: 'Degraded', variant: 'warning' },
}

interface StatusBadgeProps {
  status: StatusVariant
  className?: string
  pulse?: boolean
}

export function StatusBadge({ status, className, pulse }: StatusBadgeProps) {
  const config = variantMap[status]

  return (
    <Badge
      variant={config.variant}
      className={cn('gap-1.5 font-medium', className)}
    >
      <span
        className={cn(
          'size-1.5 rounded-full bg-current',
          pulse && 'animate-pulse',
        )}
      />
      {config.label}
    </Badge>
  )
}
