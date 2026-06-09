import { useQueryClient } from '@tanstack/react-query'
import { Trash2, UserPlus } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { useGallery } from '@/hooks/use-gallery'
import { deleteGalleryPerson, getUploadUrl, postGalleryPerson } from '@/services/api'
import { queryKeys } from '@/services/query-keys'

export function GalleryPage() {
  const { data, isLoading, error } = useGallery()
  const queryClient = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleAdd = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file || !name.trim()) {
      toast.error('Provide a name and photo.')
      return
    }
    setSubmitting(true)
    try {
      const form = new FormData()
      form.append('person_name', name.trim())
      form.append('image', file)
      await postGalleryPerson(form)
      toast.success(`${name} added to gallery`)
      setName('')
      if (fileRef.current) fileRef.current.value = ''
      queryClient.invalidateQueries({ queryKey: queryKeys.gallery })
    } catch {
      toast.error('Failed to add person')
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (id: number, personName: string) => {
    try {
      await deleteGalleryPerson(id)
      toast.success(`Removed ${personName}`)
      queryClient.invalidateQueries({ queryKey: queryKeys.gallery })
    } catch {
      toast.error('Failed to delete')
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Face Gallery</h1>
        <p className="text-sm text-muted-foreground">
          Manage authorized persons in your private recognition database.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <UserPlus className="size-5" />
            Add person
          </CardTitle>
          <CardDescription>Upload a clear front-facing photo.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1 space-y-2">
            <label className="text-sm font-medium" htmlFor="gallery-name">
              Name
            </label>
            <input
              id="gallery-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
            />
          </div>
          <input ref={fileRef} type="file" accept="image/*" className="text-sm" />
          <Button onClick={handleAdd} disabled={submitting}>
            Add to gallery
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Gallery ({data?.count ?? 0})</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading && <LoadingSpinner label="Loading gallery…" />}
          {error && (
            <p className="text-sm text-destructive">Failed to load gallery.</p>
          )}
          {data && data.items.length === 0 && (
            <p className="text-sm text-muted-foreground">No persons enrolled yet.</p>
          )}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {data?.items.map((person) => (
              <div
                key={person.id}
                className="flex items-center gap-3 rounded-lg border border-border p-3"
              >
                {person.image_path ? (
                  <img
                    src={getUploadUrl(person.image_path)}
                    alt={person.person_name}
                    className="size-16 rounded-md object-cover"
                  />
                ) : (
                  <div className="flex size-16 items-center justify-center rounded-md bg-muted text-xs">
                    No image
                  </div>
                )}
                <div className="flex-1">
                  <p className="font-medium">{person.person_name}</p>
                  <p className="text-xs text-muted-foreground">
                    Added {new Date(person.created_at).toLocaleDateString()}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(person.id, person.person_name)}
                >
                  <Trash2 className="size-4 text-destructive" />
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
