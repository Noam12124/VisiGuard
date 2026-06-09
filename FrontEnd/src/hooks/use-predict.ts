import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { postPredict } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function usePredict() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: postPredict,
    onSuccess: (data) => {
      toast.success('Prediction request sent', {
        description: data.status,
      })
      queryClient.invalidateQueries({ queryKey: queryKeys.health })
      queryClient.invalidateQueries({ queryKey: queryKeys.alerts })
      queryClient.invalidateQueries({ queryKey: queryKeys.recognized })
    },
  })
}
