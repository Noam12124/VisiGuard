import { MutationCache, QueryCache, QueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { ApiError } from '@/services/api'

function handleApiError(error: unknown) {
  if (error instanceof ApiError) {
    toast.error('API request failed', {
      description: error.message,
    })
    return
  }

  if (error instanceof Error) {
    toast.error('Something went wrong', {
      description: error.message,
    })
    return
  }

  toast.error('An unexpected error occurred')
}

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: handleApiError,
  }),
  mutationCache: new MutationCache({
    onError: handleApiError,
  }),
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      retry: 2,
    },
  },
})
