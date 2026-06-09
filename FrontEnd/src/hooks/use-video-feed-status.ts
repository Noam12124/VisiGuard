import { useQuery } from '@tanstack/react-query'
import { getVideoFeedStatus } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useVideoFeedStatus() {
  return useQuery({
    queryKey: queryKeys.videoFeedStatus,
    queryFn: getVideoFeedStatus,
    refetchInterval: 5_000,
  })
}
