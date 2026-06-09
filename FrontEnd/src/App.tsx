import { Route, Routes } from 'react-router-dom'
import { ProtectedRoute, PublicOnlyRoute } from '@/components/auth/ProtectedRoute'
import { AppShell } from '@/components/layout/AppShell'
import { AlertsPage } from '@/pages/AlertsPage'
import { CamerasPage } from '@/pages/CamerasPage'
import { ComparisonPage } from '@/pages/ComparisonPage'
import { DashboardPage } from '@/pages/DashboardPage'
import { GalleryPage } from '@/pages/GalleryPage'
import { LoginPage } from '@/pages/LoginPage'
import { MonitoringPage } from '@/pages/MonitoringPage'
import { NotFoundPage } from '@/pages/NotFoundPage'
import { RegisterPage } from '@/pages/RegisterPage'
import { SecurityAlertsPage } from '@/pages/SecurityAlertsPage'

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<PublicOnlyRoute />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
      </Route>

      <Route element={<ProtectedRoute />}>
        <Route element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="monitoring" element={<MonitoringPage />} />
          <Route path="gallery" element={<GalleryPage />} />
          <Route path="security" element={<SecurityAlertsPage />} />
          <Route path="cameras" element={<CamerasPage />} />
          <Route path="compare" element={<ComparisonPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Route>
    </Routes>
  )
}
