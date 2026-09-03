import React from "react"
import { Link, useLocation } from "react-router-dom"
import { useQuery } from "@tanstack/react-query"
import {
  Activity,
  Bot,
  FileCheck2,
  LayoutDashboard,
  Moon,
  Sparkles,
  Sun,
  Upload,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { apiClient } from "@/lib/api-client"
import { useAnalysisStore } from "@/store/useAnalysisStore"
import { useAppStore } from "@/store/useAppStore"

export const Navbar: React.FC = () => {
  const location = useLocation()
  const { theme, toggleTheme, setSystemStatus } = useAppStore()
  const { isAnalyzing } = useAnalysisStore()

  const { data: status } = useQuery({
    queryKey: ["systemStatus"],
    queryFn: async () => {
      const data = await apiClient.getSystemStatus()
      setSystemStatus(data)
      return data
    },
    refetchInterval: 15000,
  })

  const navItems = [
    { path: "/", label: "Overview", icon: Sparkles },
    { path: "/upload", label: "Upload", icon: Upload },
    { path: "/analysis", label: "Analysis", icon: Activity },
    { path: "/results", label: "Results", icon: FileCheck2 },
    { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { path: "/chat", label: "AI Assistant", icon: Bot },
  ]

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="container mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-3 transition-opacity hover:opacity-90">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-emerald-500 shadow-lg shadow-cyan-500/20">
            <span className="text-xl font-black text-slate-950">OV</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight text-white">OncoVision AI</span>
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            <p className="text-xs text-slate-400">Clinical Screening Workspace</p>
          </div>
        </Link>

        {/* Center Nav */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800/90 shadow-inner">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? "bg-cyan-500/15 text-cyan-400 shadow-sm border border-cyan-500/30"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </Link>
            )
          })}
        </nav>

        {/* Right Badges & Controls */}
        <div className="flex items-center gap-3">
          {/* Status Badges */}
          <div className="hidden lg:flex items-center gap-2">
            <Badge variant={status?.models_loaded ? "success" : "secondary"}>
              Models: {status?.models_loaded ? "Ready" : "Standby"}
            </Badge>
            <Badge variant={status?.llm_connected ? "default" : "secondary"}>
              LLM: {status?.llm_connected ? "OpenRouter" : "Offline"}
            </Badge>
            {isAnalyzing && (
              <Badge variant="warning" className="animate-pulse">
                Analyzing...
              </Badge>
            )}
          </div>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={toggleTheme}
            className="text-slate-400 hover:text-white"
            title="Toggle theme"
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  )
}
