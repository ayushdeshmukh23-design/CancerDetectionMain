import { create } from "zustand"
import type { SystemStatus } from "@/types/diagnosis"

interface AppState {
  theme: "dark" | "light"
  systemStatus: SystemStatus | null
  toggleTheme: () => void
  setSystemStatus: (status: SystemStatus) => void
}

export const useAppStore = create<AppState>((set) => ({
  theme: "dark",
  systemStatus: null,
  toggleTheme: () =>
    set((state) => {
      const nextTheme = state.theme === "dark" ? "light" : "dark"
      if (nextTheme === "dark") {
        document.documentElement.classList.add("dark")
      } else {
        document.documentElement.classList.remove("dark")
      }
      return { theme: nextTheme }
    }),
  setSystemStatus: (systemStatus) => set({ systemStatus }),
}))
