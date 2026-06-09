import { useQuery } from '@tanstack/react-query'
import { getSecurityEvents } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useSecurityEvents() {
  return useQuery({
    queryKey: queryKeys.securityEvents,
    queryFn: getSecurityEvents,
    refetchInterval: 3000,
  })
}
