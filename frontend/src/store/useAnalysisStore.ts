import { create } from "zustand"
import type { DiagnosisState } from "@/types/diagnosis"

interface AnalysisStoreState {
  activeJobId: string | null
  isAnalyzing: boolean
  diagnosisState: DiagnosisState | null
  lastCompleteState: DiagnosisState | null
  error: string | null
  setActiveJobId: (id: string | null) => void
  setIsAnalyzing: (analyzing: boolean) => void
  setDiagnosisState: (state: DiagnosisState | null) => void
  setError: (err: string | null) => void
  reset: () => void
}

export const useAnalysisStore = create<AnalysisStoreState>((set) => ({
  activeJobId: null,
  isAnalyzing: false,
  diagnosisState: null,
  lastCompleteState: null,
  error: null,
  setActiveJobId: (activeJobId) => set({ activeJobId, isAnalyzing: !!activeJobId }),
  setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setDiagnosisState: (diagnosisState) =>
    set((prev) => ({
      diagnosisState,
      lastCompleteState:
        diagnosisState && diagnosisState.ensemble_result ? diagnosisState : prev.lastCompleteState,
      isAnalyzing: false,
      activeJobId: null,
    })),
  setError: (error) => set({ error, isAnalyzing: false, activeJobId: null }),
  reset: () => set({ activeJobId: null, isAnalyzing: false, diagnosisState: null, error: null }),
}))
