import { useQuery } from '@tanstack/react-query'
import { getRecognized } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

const POLL_MS = 3_000

export function useRecognized() {
  return useQuery({
    queryKey: queryKeys.recognized,
    queryFn: getRecognized,
    refetchInterval: POLL_MS,
  })
}
