import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  clearAuth,
  getStoredUser,
  setAuth,
  type StoredUser,
} from '@/lib/auth-storage'
import { postLogin, postSignUp } from '@/services/api'
import type { LoginRequest, SignUpRequest } from '@/types/auth'

interface AuthContextValue {
  user: StoredUser | null
  isAuthenticated: boolean
  login: (body: LoginRequest) => Promise<void>
  signup: (body: SignUpRequest) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<StoredUser | null>(() => getStoredUser())

  const login = useCallback(async (body: LoginRequest) => {
    const res = await postLogin(body)
    setAuth(res.access_token, res.user)
    setUser(res.user)
  }, [])

  const signup = useCallback(async (body: SignUpRequest) => {
    const res = await postSignUp(body)
    setAuth(res.access_token, res.user)
    setUser(res.user)
  }, [])

  const logout = useCallback(() => {
    clearAuth()
    setUser(null)
  }, [])

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: user !== null,
      login,
      signup,
      logout,
    }),
    [user, login, signup, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}
