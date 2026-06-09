import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { getCameras, postSetCamera } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useCameras() {
  return useQuery({
    queryKey: queryKeys.cameras,
    queryFn: getCameras,
    staleTime: 30_000,
  })
}

export function useSetCamera() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: postSetCamera,
    onSuccess: (data) => {
      if (data.success) {
        toast.success('Camera switched', { description: data.message })
      } else {
        toast.error('Camera switch failed', { description: data.message })
      }
      queryClient.invalidateQueries({ queryKey: queryKeys.cameras })
      queryClient.invalidateQueries({ queryKey: queryKeys.videoFeedStatus })
    },
  })
}
