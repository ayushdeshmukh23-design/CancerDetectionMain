import { create } from "zustand"
import type { ImageUploadResponse, ThermalUploadResponse } from "@/types/diagnosis"

interface UploadState {
  image: ImageUploadResponse | null
  thermal: ThermalUploadResponse | null
  setImage: (img: ImageUploadResponse | null) => void
  setThermal: (th: ThermalUploadResponse | null) => void
  reset: () => void
}

export const useUploadStore = create<UploadState>((set) => ({
  image: null,
  thermal: null,
  setImage: (image) => set({ image }),
  setThermal: (thermal) => set({ thermal }),
  reset: () => set({ image: null, thermal: null }),
}))
