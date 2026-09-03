import type {
  ChatMessage,
  DiagnosisState,
  HistoricalPoint,
  ImageUploadResponse,
  SystemStatus,
  ThermalUploadResponse,
} from "@/types/diagnosis"

const API_BASE = ""

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = "ApiError"
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = "API request failed"
    try {
      const data = await res.json()
      errorDetail = data.detail || data.message || errorDetail
    } catch {
      errorDetail = await res.text() || res.statusText
    }
    throw new ApiError(res.status, errorDetail)
  }
  return res.json()
}

export const apiClient = {
  async getSystemStatus(): Promise<SystemStatus> {
    const res = await fetch(`${API_BASE}/api/system/status`)
    return handleResponse<SystemStatus>(res)
  },

  async uploadImage(file: File): Promise<ImageUploadResponse> {
    const formData = new FormData()
    formData.append("file", file)
    const res = await fetch(`${API_BASE}/api/upload/image`, {
      method: "POST",
      body: formData,
    })
    return handleResponse<ImageUploadResponse>(res)
  },

  async uploadThermal(file: File): Promise<ThermalUploadResponse> {
    const formData = new FormData()
    formData.append("file", file)
    const res = await fetch(`${API_BASE}/api/upload/thermal`, {
      method: "POST",
      body: formData,
    })
    return handleResponse<ThermalUploadResponse>(res)
  },

  async startAnalysis(
    imagePath: string,
    thermalPath?: string | null,
    patientContext?: string
  ): Promise<{ success: boolean; job_id: string }> {
    const res = await fetch(`${API_BASE}/api/analysis/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_path: imagePath,
        thermal_matrix_path: thermalPath || null,
        patient_context: patientContext || "",
      }),
    })
    return handleResponse<{ success: boolean; job_id: string }>(res)
  },

  async getAnalysisStatus(jobId: string): Promise<{
    job_id: string
    status: "running" | "done" | "failed"
    error?: string | null
    result?: DiagnosisState | null
    elapsed: number
  }> {
    const res = await fetch(`${API_BASE}/api/analysis/status/${jobId}`)
    return handleResponse(res)
  },

  async getDemoAnalysis(caseId: string = "case_malignant"): Promise<{
    diagnosis_state: DiagnosisState
    seeded_history: HistoricalPoint[]
  }> {
    const res = await fetch(`${API_BASE}/api/analysis/demo?case_id=${encodeURIComponent(caseId)}`)
    return handleResponse(res)
  },

  async exportReport(
    reportMarkdown: string,
    format: "pdf" | "md" = "pdf"
  ): Promise<{ success: boolean; filename: string; download_url: string }> {
    const res = await fetch(`${API_BASE}/api/reports/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report_markdown: reportMarkdown,
        format,
      }),
    })
    return handleResponse(res)
  },

  async exportChat(
    history: ChatMessage[]
  ): Promise<{ success: boolean; filename: string; download_url: string }> {
    const res = await fetch(`${API_BASE}/api/chat/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        history: history.map((h) => ({ role: h.role, content: h.content })),
      }),
    })
    return handleResponse(res)
  },

  streamChat(
    prompt: string,
    reportContext: string,
    history: ChatMessage[],
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: Error) => void,
    model?: string,
    webSearch?: boolean
  ) {
    const controller = new AbortController()

    fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        report_context: reportContext,
        history: history.map((m) => ({ role: m.role, content: m.content })),
        model: model || undefined,
        web_search: Boolean(webSearch),
      }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Chat request failed with status ${res.status}`)
        }
        const reader = res.body?.getReader()
        if (!reader) throw new Error("No response body")

        const decoder = new TextDecoder()
        let buffer = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n\n")
          buffer = lines.pop() || ""

          for (const line of lines) {
            const trimmed = line.trim()
            if (trimmed.startsWith("data:")) {
              const dataStr = trimmed.slice(5).trim()
              if (dataStr === "[DONE]") {
                onDone()
                return
              }
              try {
                const parsed = JSON.parse(dataStr)
                if (parsed.token) onToken(parsed.token)
                if (parsed.error) onError(new Error(parsed.error))
              } catch {
                // ignore
              }
            }
          }
        }
        onDone()
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          onError(err)
        }
      })

    return () => controller.abort()
  },
}
