import { useQuery } from '@tanstack/react-query'
import { getHealth } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useHealthCheck() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: getHealth,
    refetchInterval: 30_000,
  })
}
