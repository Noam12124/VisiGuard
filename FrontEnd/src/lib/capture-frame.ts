import { getApiBaseUrl } from '@/services/api-client'

/** Fetch a compact JPEG snapshot from the backend (fast, no canvas work). */
export async function captureFrameFromApi(): Promise<Blob | null> {
  try {
    const res = await fetch(`${getApiBaseUrl()}/video-feed/snapshot?compact=true`)
    if (!res.ok) return null
    return res.blob()
  } catch {
    return null
  }
}

/** Capture a JPEG blob from a live MJPEG <img> element. */
export async function captureFrameFromImage(
  img: HTMLImageElement,
  quality = 0.85,
): Promise<Blob | null> {
  if (!img.naturalWidth || !img.naturalHeight) {
    console.warn('[capture-frame] Image not ready:', {
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
      complete: img.complete,
    })
    return null
  }

  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    console.error('[capture-frame] Could not get 2d context')
    return null
  }

  try {
    ctx.drawImage(img, 0, 0)
  } catch (err) {
    console.error('[capture-frame] drawImage failed (CORS?):', err)
    return null
  }

  return new Promise((resolve) => {
    try {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            console.error('[capture-frame] toBlob returned null (tainted canvas?)')
          } else {
            console.log('[capture-frame] Captured frame:', {
              width: canvas.width,
              height: canvas.height,
              bytes: blob.size,
              type: blob.type,
            })
          }
          resolve(blob)
        },
        'image/jpeg',
        quality,
      )
    } catch (err) {
      console.error('[capture-frame] toBlob failed:', err)
      resolve(null)
    }
  })
}
