import React from "react"
import { BrowserRouter, Route, Routes } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Toaster } from "sonner"
import { AppLayout } from "@/components/layout/AppLayout"
import { OverviewPage } from "@/features/overview/OverviewPage"
import { UploadPage } from "@/features/upload/UploadPage"
import { AnalysisPage } from "@/features/analysis/AnalysisPage"
import { ResultsPage } from "@/features/results/ResultsPage"
import { DashboardPage } from "@/features/dashboard/DashboardPage"
import { ChatPage } from "@/features/chat/ChatPage"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 1000 * 30, // 30 seconds
    },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" theme="dark" richColors closeButton />
    </QueryClientProvider>
  )
}

export default App
