import { useQuery } from '@tanstack/react-query'
import { getGallery } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function useGallery() {
  return useQuery({
    queryKey: queryKeys.gallery,
    queryFn: getGallery,
  })
}
