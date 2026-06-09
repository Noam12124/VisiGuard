import { LogOut, Menu, Shield } from 'lucide-react'
import { StatusBadge } from '@/components/common/StatusBadge'
import { Button } from '@/components/ui/button'
import { LiveMonitoringIndicator } from '@/components/surveillance/LiveMonitoringIndicator'
import { useAuth } from '@/context/AuthContext'
import { useMonitoringContext } from '@/context/MonitoringContext'
import { useHealthCheck } from '@/hooks/use-health-check'

interface HeaderProps {
  onMenuClick?: () => void
}

export function Header({ onMenuClick }: HeaderProps) {
  const { isLoading, isError, isSuccess } = useHealthCheck()
  const { user, logout } = useAuth()
  const { enabled, setEnabled, monitoringStatus } = useMonitoringContext()

  const status = isLoading
    ? 'pending'
    : isError
      ? 'offline'
      : isSuccess
        ? 'online'
        : 'warning'

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/80 px-4 backdrop-blur-md md:px-6">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={onMenuClick}
          aria-label="Open navigation menu"
        >
          <Menu className="size-5" />
        </Button>

        <div className="flex items-center gap-2 md:hidden">
          <Shield className="size-5 text-primary" />
          <span className="font-semibold">VisiGuard</span>
        </div>

        <div className="hidden md:block">
          <h1 className="text-sm font-medium text-muted-foreground">
            Home Security Command Center
          </h1>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        <label className="hidden cursor-pointer items-center gap-2 sm:flex">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
            className="size-3.5 rounded border-border accent-primary"
            aria-label="Toggle background monitoring"
          />
          <LiveMonitoringIndicator status={monitoringStatus} compact />
        </label>
        <div className="sm:hidden">
          <LiveMonitoringIndicator status={monitoringStatus} compact />
        </div>
        {user && (
          <span className="hidden text-sm text-muted-foreground lg:inline">
            {user.username}
          </span>
        )}
        <StatusBadge status={status} pulse={status === 'online'} />
        <Button variant="outline" size="sm" onClick={logout}>
          <LogOut className="size-4" />
          <span className="hidden sm:inline">Logout</span>
        </Button>
      </div>
    </header>
  )
}
