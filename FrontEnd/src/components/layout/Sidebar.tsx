import {
  Activity,
  Bell,
  LayoutDashboard,
  ScanFace,
  Shield,
  ShieldAlert,
  Users,
  Video,
  type LucideIcon,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Separator } from '@/components/ui/separator'

interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  end?: boolean
}

const navItems: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/monitoring', label: 'Monitoring', icon: Video },
  { to: '/gallery', label: 'Gallery', icon: Users },
  { to: '/security', label: 'Security Alerts', icon: ShieldAlert },
  { to: '/cameras', label: 'Cameras', icon: Video },
  { to: '/compare', label: 'Face Comparison', icon: ScanFace },
  { to: '/alerts', label: 'Legacy Alerts', icon: Bell },
]

export function Sidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-card/50 md:flex">
      <div className="flex h-16 items-center gap-3 px-6">
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary/15 glow-emerald">
          <Shield className="size-5 text-primary" />
        </div>
        <div>
          <p className="text-sm font-semibold tracking-tight">VisiGuard</p>
          <p className="text-xs text-muted-foreground">Security Console</p>
        </div>
      </div>

      <Separator />

      <nav className="flex flex-1 flex-col gap-1 p-4">
        <p className="mb-2 px-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Navigation
        </p>
        {navItems.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )
            }
          >
            <Icon className="size-4 shrink-0" />
            <span className="flex-1">{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto border-t border-border p-4">
        <div className="rounded-lg bg-muted/50 p-3">
          <div className="flex items-center gap-2 text-xs font-medium">
            <Activity className="size-3.5 text-primary" />
            AI Pipeline
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            YOLO detection → ArcFace embedding → Cosine match
          </p>
        </div>
      </div>
    </aside>
  )
}
