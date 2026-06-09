let alarmAudio: HTMLAudioElement | null = null

/** Play intrusion alarm; falls back to Web Audio beep if mp3 missing. */
export function playIntrusionAlarm(): void {
  try {
    if (!alarmAudio) {
      alarmAudio = new Audio('/sounds/alarm.mp3')
    }
    alarmAudio.currentTime = 0
    void alarmAudio.play().catch(() => playBeepFallback())
  } catch {
    playBeepFallback()
  }
}

function playBeepFallback(): void {
  try {
    const ctx = new AudioContext()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'square'
    osc.frequency.value = 880
    gain.gain.value = 0.15
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start()
    setTimeout(() => {
      osc.stop()
      void ctx.close()
    }, 400)
  } catch {
    /* silent if audio blocked */
  }
}
