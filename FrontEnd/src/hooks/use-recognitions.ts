import { useQuery } from '@tanstack/react-query'
import { getRecognitions } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useRecognitions() {
  return useQuery({
    queryKey: queryKeys.recognitions,
    queryFn: getRecognitions,
    refetchInterval: 15000,
  })
}
