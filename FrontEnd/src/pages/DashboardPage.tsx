import { Link } from 'react-router-dom'
import {
  Activity,
  Bell,
  ScanFace,
  Users,
  Video,
} from 'lucide-react'
import { DashboardOverview } from '@/components/dashboard/DashboardOverview'
import { SystemStatusCard } from '@/components/dashboard/SystemStatusCard'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { useAuth } from '@/context/AuthContext'
import { useGallery } from '@/hooks/use-gallery'
import { useRecognitions } from '@/hooks/use-recognitions'
import { useSecurityEvents } from '@/hooks/use-security-events'
import { useUnknowns } from '@/hooks/use-unknowns'

export function DashboardPage() {
  const { user } = useAuth()
  const { data: gallery } = useGallery()
  const { data: recognitions } = useRecognitions()
  const { data: unknowns } = useUnknowns()
  const { data: securityEvents } = useSecurityEvents()

  const pendingUnknowns =
    unknowns?.items.filter((u) => !u.action_taken).length ?? 0

  return (
    <div className="space-y-6">
      <DashboardOverview />

      <div className="rounded-lg border border-border bg-card/50 px-4 py-3">
        <p className="text-sm">
          Signed in as <span className="font-medium text-primary">{user?.username}</span>
          {' — '}
          your gallery and monitoring data are private to your account.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Gallery persons</CardDescription>
            <CardTitle className="text-3xl">{gallery?.count ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Recognitions</CardDescription>
            <CardTitle className="text-3xl">{recognitions?.count ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Pending unknowns</CardDescription>
            <CardTitle className="text-3xl">{pendingUnknowns}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Security events</CardDescription>
            <CardTitle className="text-3xl">{securityEvents?.count ?? 0}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <Video className="mb-2 size-8 text-primary" />
            <CardTitle>Live Monitoring</CardTitle>
            <CardDescription>
              Webcam stream with YOLO detection and ArcFace recognition every 3 seconds.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild>
              <Link to="/monitoring">Open monitoring</Link>
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Users className="mb-2 size-8 text-primary" />
            <CardTitle>Face Gallery</CardTitle>
            <CardDescription>
              Enroll and manage authorized persons in your private gallery.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline">
              <Link to="/gallery">Manage gallery</Link>
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Bell className="mb-2 size-8 text-primary" />
            <CardTitle>Security Alerts</CardTitle>
            <CardDescription>
              Review alarm triggers and unknown person resolutions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline">
              <Link to="/security">View alerts</Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SystemStatusCard />
        <Card>
          <CardHeader>
            <Activity className="mb-2 size-8 text-primary" />
            <CardTitle>Quick tools</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button asChild variant="outline" size="sm">
              <Link to="/cameras">
                <Video className="size-4" />
                Cameras
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link to="/compare">
                <ScanFace className="size-4" />
                Compare faces
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link to="/alerts">
                <Bell className="size-4" />
                Legacy alerts
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
