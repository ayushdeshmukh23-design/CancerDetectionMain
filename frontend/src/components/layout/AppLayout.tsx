import React from "react"
import { Outlet } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { Navbar } from "./Navbar"
import { ShieldAlert } from "lucide-react"

export const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-300">
      <Navbar />

      <main className="flex-1 container mx-auto max-w-7xl px-4 sm:px-6 py-6 sm:py-8">
        <AnimatePresence mode="wait">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Medical Disclaimer Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/90 py-4 text-center text-xs text-slate-500">
        <div className="container mx-auto flex items-center justify-center gap-2 px-4">
          <ShieldAlert className="h-4 w-4 text-amber-500/80 shrink-0" />
          <span>
            <strong>Important Medical Disclaimer:</strong> OncoVision AI is a decision-support research prototype and not a standalone medical diagnostic device. All outputs must be validated by qualified clinicians.
          </span>
        </div>
      </footer>
    </div>
  )
}
