import { useMutation } from '@tanstack/react-query'
import { postCompareFaces } from '@/services/api'

export function useCompareFaces() {
  return useMutation({
    mutationFn: (formData: FormData) => postCompareFaces(formData),
  })
}
