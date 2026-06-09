import { useQuery } from '@tanstack/react-query'
import { getUnknowns } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useUnknowns() {
  return useQuery({
    queryKey: queryKeys.unknowns,
    queryFn: getUnknowns,
    refetchInterval: 15000,
  })
}
