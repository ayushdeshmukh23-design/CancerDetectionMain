import React from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  ArrowRight,
  Bot,
  BrainCircuit,
  FileCheck2,
  Flame,
  Gauge,
  Layers,
  LayoutDashboard,
  Microscope,
  ScanEye,
  ShieldAlert,
  Sparkles,
  ThermometerSnowflake,
  Upload,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { apiClient } from "@/lib/api-client"
import { useAnalysisStore } from "@/store/useAnalysisStore"

export const OverviewPage: React.FC = () => {
  const navigate = useNavigate()
  const { setDiagnosisState } = useAnalysisStore()

  const handleLoadDemo = async (caseId: string = "case_malignant") => {
    try {
      const data = await apiClient.getDemoAnalysis(caseId)
      setDiagnosisState(data.diagnosis_state)
      navigate("/results")
    } catch (err) {
      console.error("Failed to load demo:", err)
    }
  }

  const capabilities = [
    {
      title: "Dual-Modality Co-Registration & Synoptic Fusion",
      desc: "Co-registers high-resolution Full-Field Digital Mammography (FFDM) with sub-millikelvin calibrated infrared thermography matrices for synchronized structural and metabolic evaluation.",
      icon: Flame,
      color: "text-amber-400 bg-amber-500/10 border-amber-500/20",
      badge: "Core Architecture",
    },
    {
      title: "Multi-Model Deep Ensemble Learning",
      desc: "Synthesizes multi-scale EfficientNet-B4 CNN backbones, Vision Transformer (ViT-B/16) tokenized spatial attention, and calibrated XGBoost multimodal metaclassifiers.",
      icon: BrainCircuit,
      color: "text-cyan-400 bg-cyan-500/10 border-cyan-500/20",
      badge: "Neural Ensemble",
    },
    {
      title: "Multi-Technique Explainable AI (XAI) Suite",
      desc: "Rigorous evidence localization powered by Grad-CAM++ colormaps, ViT patch self-attention rollout, LIME superpixel perturbations, Integrated Gradients, and TreeSHAP waterfall attributions.",
      icon: Layers,
      color: "text-purple-400 bg-purple-500/10 border-purple-500/20",
      badge: "Transparency",
    },
    {
      title: "Autonomous ACR BI-RADS & TNM Triage Classifier",
      desc: "Algorithmic decision logic mapping multimodal posterior probabilities directly into standardized ACR BI-RADS categories (1 to 5) with priority clinical management protocols.",
      icon: ShieldAlert,
      color: "text-rose-400 bg-rose-500/10 border-rose-500/20",
      badge: "Clinical Triage",
    },
    {
      title: "Sub-Millikelvin Thermal Asymmetry Quantification (ΔT)",
      desc: "Automated contra-lateral quadrant thermic gradient extraction detecting focal hyperthermic emission (ΔT > +1.5°C) correlated with tumor neo-angiogenesis and metabolic heat flux.",
      icon: ThermometerSnowflake,
      color: "text-orange-400 bg-orange-500/10 border-orange-500/20",
      badge: "Physiological Biomarker",
    },
    {
      title: "Dynamic Lesion Semantic Segmentation (U-Net)",
      desc: "Deep convolutional contouring isolating hyperdense parenchymal masses, spiculation margins, and microcalcification clusters with quantified Dice similarity (>0.94).",
      icon: Microscope,
      color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
      badge: "Morphology",
    },
    {
      title: "Bayesian Uncertainty & Temperature Calibration",
      desc: "Multi-tier epistemic and aleatoric uncertainty decomposition using temperature scaling, providing 95% confidence intervals and entropy dispersion metrics to curb overconfidence.",
      icon: Gauge,
      color: "text-blue-400 bg-blue-500/10 border-blue-500/20",
      badge: "Reliability Engineering",
    },
    {
      title: "OpenRouter Clinical Copilot & 11-Section Reporting",
      desc: "Zero-latency clinical markdown reports synthesized through state-of-the-art LLMs, featuring patient plain-language translations and interactive streaming chat assistance.",
      icon: Bot,
      color: "text-teal-400 bg-teal-500/10 border-teal-500/20",
      badge: "Clinical LLM",
    },
  ]

  const workflowSteps = [
    {
      step: "01",
      name: "Upload & Validate",
      icon: Upload,
      path: "/upload",
      desc: "Ingest DICOM/FFDM mammograms with optional FLIR radiometric thermal matrices, with instant SNR/CNR checks and benchmark presets.",
    },
    {
      step: "02",
      name: "AI Pipeline Analysis",
      icon: Activity,
      path: "/analysis",
      desc: "Inspect the 6-stage transformation matrix, hardware compute allocation, latency waterfall, and real-time execution audit logs.",
    },
    {
      step: "03",
      name: "Results & Triage",
      icon: FileCheck2,
      path: "/results",
      desc: "Evaluate calibrated BI-RADS triage, model concordance, all 7 XAI evidence localizations with opacity blend, and download PDF reports.",
    },
    {
      step: "04",
      name: "Longitudinal Dashboard",
      icon: LayoutDashboard,
      path: "/dashboard",
      desc: "Track tumor doubling kinetics, contra-lateral ΔT heat flux, calibration curves, and 25-point patient progression.",
    },
    {
      step: "05",
      name: "Clinical Assistant",
      icon: Bot,
      path: "/chat",
      desc: "Conduct interactive clinical consultations grounded in multimodal patient findings with live token streaming.",
    },
  ]

  return (
    <div className="space-y-12">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900/90 via-slate-900/60 to-slate-950 p-8 sm:p-12 shadow-2xl">
        <div className="absolute top-0 right-0 -mr-20 -mt-20 h-96 w-96 rounded-full bg-cyan-500/15 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -ml-20 -mb-20 h-96 w-96 rounded-full bg-purple-500/15 blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-4xl space-y-6">
          <div className="flex flex-wrap items-center gap-2.5">
            <Badge variant="default" className="text-xs px-3 py-1 font-semibold tracking-wide">
              <Sparkles className="mr-1.5 h-3.5 w-3.5" /> Next-Generation Oncological Diagnostic Workstation
            </Badge>
            <Badge variant="secondary" className="text-xs px-3 py-1 font-mono">
              v2.4 Enterprise Release
            </Badge>
            <Badge variant="outline" className="text-xs border-cyan-500/40 text-cyan-400">
              ACR BI-RADS & Dual-Modality Fusion
            </Badge>
          </div>

          <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-tight">
            Multi-Modal Breast Cancer Screening & Explainable AI Workstation
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed font-normal">
            A state-of-the-art oncological clinical workstation engineered for synoptic co-registration of Full-Field Digital Mammography (FFDM) with sub-millikelvin calibrated infrared thermographic matrices. Delivers calibrated, evidence-backed malignancy risk stratification through ensemble deep neural architectures (EfficientNet-B4 + Vision Transformer + XGBoost Metaclassifiers), multi-layered Explainable AI (Grad-CAM++, ViT Attention Rollout, TreeSHAP, LIME Superpixels, & Thermal Isotherm Delta-T maps), and real-time interactive OpenRouter clinical LLM assistance.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <Button size="lg" onClick={() => navigate("/upload")} className="gap-2 shadow-lg shadow-cyan-500/20">
              <Upload className="h-4 w-4" /> Launch Screening Ingestion
            </Button>
            <Button variant="outline" size="lg" onClick={() => handleLoadDemo("case_malignant")} className="gap-2">
              <ScanEye className="h-4 w-4 text-rose-400" /> Pre-load High-Risk Malignant Demo (BI-RADS 4C)
            </Button>
            <Button
              variant="ghost"
              size="lg"
              onClick={() => handleLoadDemo("case_benign")}
              className="gap-2 text-slate-400 hover:text-white"
            >
              Load Benign Case (BI-RADS 3)
            </Button>
          </div>
        </div>
      </div>

      {/* 8 System Capabilities Grid */}
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <BrainCircuit className="h-6 w-6 text-cyan-400" /> Core System Capabilities
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Engineered to meet rigorous clinical diagnostic standards across multi-modal vision, thermography, and explainability.
            </p>
          </div>
          <Badge variant="secondary" className="self-start sm:self-auto font-mono text-xs">
            8 Advanced Modules Active
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {capabilities.map((cap, i) => {
            const Icon = cap.icon
            return (
              <Card
                key={i}
                className="border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900/90 transition-all group flex flex-col justify-between"
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className={`p-2.5 rounded-xl border ${cap.color} transition-transform group-hover:scale-105`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <Badge variant="outline" className="text-[10px] text-slate-400 border-slate-700">
                      {cap.badge}
                    </Badge>
                  </div>
                  <CardTitle className="text-base font-bold text-slate-100 group-hover:text-white leading-snug">
                    {cap.title}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <CardDescription className="text-xs text-slate-400 leading-relaxed font-normal">
                    {cap.desc}
                  </CardDescription>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Interactive Workflow Navigator */}
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <Activity className="h-6 w-6 text-cyan-400" /> Clinical Diagnostic Workflow
          </h2>
          <p className="text-slate-400 text-sm mt-1">
            Five-stage pipeline guiding clinicians from multi-modal ingestion to explainable evidence audit and longitudinal telemetry.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {workflowSteps.map((ws) => {
            const Icon = ws.icon
            return (
              <div
                key={ws.step}
                onClick={() => navigate(ws.path)}
                className="cursor-pointer group relative overflow-hidden rounded-xl border border-slate-800 bg-slate-900/50 hover:bg-slate-900/90 hover:border-cyan-500/40 p-5 transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
                      {ws.step}
                    </span>
                    <Icon className="h-4 w-4 text-slate-500 group-hover:text-cyan-400 transition-colors" />
                  </div>
                  <h3 className="font-bold text-sm text-slate-200 group-hover:text-white">{ws.name}</h3>
                  <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{ws.desc}</p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center text-xs font-semibold text-cyan-400 group-hover:translate-x-1 transition-transform">
                  <span>Enter Stage</span>
                  <ArrowRight className="h-3 w-3 ml-1" />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
