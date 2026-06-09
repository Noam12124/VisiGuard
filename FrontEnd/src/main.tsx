import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'sonner'
import { AppRoutes } from '@/App'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { AuthProvider } from '@/context/AuthContext'
import { queryClient } from '@/lib/query-client'
import '@/index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <BrowserRouter>
            <AppRoutes />
            <Toaster
              theme="dark"
              position="top-right"
              richColors
              closeButton
              toastOptions={{
                classNames: {
                  toast: 'border-border bg-card text-foreground',
                },
              }}
            />
          </BrowserRouter>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
)
