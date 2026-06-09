import { useQuery } from '@tanstack/react-query'
import { getAlerts } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

const POLL_MS = 3_000

export function useAlerts() {
  return useQuery({
    queryKey: queryKeys.alerts,
    queryFn: getAlerts,
    refetchInterval: POLL_MS,
  })
}
