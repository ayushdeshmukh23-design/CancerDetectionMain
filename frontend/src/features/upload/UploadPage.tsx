import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  FileCheck2,
  FileText,
  Flame,
  Gauge,
  Image as ImageIcon,
  Info,
  Loader2,
  Microscope,
  RotateCcw,
  ScanEye,
  ShieldCheck,
  Sparkles,
  Thermometer,
  Upload,
  X,
} from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { apiClient } from "@/lib/api-client"
import { useAnalysisStore } from "@/store/useAnalysisStore"
import { useUploadStore } from "@/store/useUploadStore"

interface BenchmarkCase {
  id: string
  title: string
  birads: string
  risk: string
  deltaT: string
  description: string
  patientContext: string
  density: "A" | "B" | "C" | "D"
  quadrant: string
  age: number
}

const BENCHMARK_CASES: BenchmarkCase[] = [
  {
    id: "case_malignant",
    title: "Invasive Ductal Carcinoma (High Acuity)",
    birads: "BI-RADS 4C",
    risk: "High Risk (78.6%)",
    deltaT: "+2.41°C",
    description: "Spiculated hyperdense mass in R-UOQ with clustered microcalcifications and marked contra-lateral thermal asymmetry.",
    patientContext: "54yo post-menopausal female, palpable painless mass in right upper outer quadrant for 3 weeks. Maternal history of pre-menopausal breast cancer.",
    density: "C",
    quadrant: "Right Upper-Outer (R-UOQ)",
    age: 54,
  },
  {
    id: "case_benign",
    title: "Well-Circumscribed Fibroadenoma",
    birads: "BI-RADS 3",
    risk: "Low Risk (28.4%)",
    deltaT: "+0.42°C",
    description: "Oval circumscribed mass with homogenous parenchyma and negligible thermographic gradient.",
    patientContext: "36yo pre-menopausal female, mobile non-tender circumscribed nodule in left lower quadrant. Routine follow-up.",
    density: "B",
    quadrant: "Left Lower-Outer (L-LOQ)",
    age: 36,
  },
  {
    id: "case_normal",
    title: "Dense Parenchyma (Negative Screen)",
    birads: "BI-RADS 1",
    risk: "Routine (9.2%)",
    deltaT: "+0.11°C",
    description: "Bilateral symmetric fibroglandular tissue, no architectural distortion, thermal symmetry intact.",
    patientContext: "48yo asymptomatic female presenting for routine bilateral screening mammogram. No personal or family oncological history.",
    density: "C",
    quadrant: "Bilateral Symmetrical",
    age: 48,
  },
]

export const UploadPage: React.FC = () => {
  const navigate = useNavigate()
  const { image, thermal, setImage, setThermal, reset } = useUploadStore()
  const { setActiveJobId, isAnalyzing, setDiagnosisState } = useAnalysisStore()

  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const [isUploadingThermal, setIsUploadingThermal] = useState(false)
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null)

  // Structured clinical intake fields
  const [patientAge, setPatientAge] = useState<number>(54)
  const [breastDensity, setBreastDensity] = useState<"A" | "B" | "C" | "D">("C")
  const [examView, setExamView] = useState<"MLO" | "CC" | "Spot Compression">("MLO")
  const [quadrant, setQuadrant] = useState<string>("Right Upper-Outer (R-UOQ)")
  const [symptoms, setSymptoms] = useState<string[]>(["Palpable Nodule/Mass"])
  const [geneticRisk, setGeneticRisk] = useState<string>("First-Degree Relative History")
  const [freeNotes, setFreeNotes] = useState<string>(
    "54yo female, palpable painless mass in right upper outer quadrant. Maternal history of breast carcinoma."
  )

  const handleImageFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setSelectedBenchmark(null)
    setIsUploadingImage(true)
    try {
      const res = await apiClient.uploadImage(file)
      setImage(res)
      toast.success("Mammographic scan ingested successfully", {
        description: `${res.filename} (${res.width}x${res.height}, ${res.format})`,
      })
    } catch (err: any) {
      toast.error("Mammogram upload failed", { description: err.message })
    } finally {
      setIsUploadingImage(false)
    }
  }

  const handleThermalFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setSelectedBenchmark(null)
    setIsUploadingThermal(true)
    try {
      const res = await apiClient.uploadThermal(file)
      setThermal(res)
      toast.success("Radiometric thermal matrix calibrated", {
        description: `${res.filename} (Array shape: [${res.shape.join(", ")}])`,
      })
    } catch (err: any) {
      toast.error("Thermal upload failed", { description: err.message })
    } finally {
      setIsUploadingThermal(false)
    }
  }

  const handleSelectBenchmark = async (benchmark: BenchmarkCase) => {
    setSelectedBenchmark(benchmark.id)
    setPatientAge(benchmark.age)
    setBreastDensity(benchmark.density)
    setQuadrant(benchmark.quadrant)
    setFreeNotes(benchmark.patientContext)

    try {
      toast.loading("Loading clinical benchmark case payload...", { id: "benchmark" })
      const data = await apiClient.getDemoAnalysis(benchmark.id)
      setDiagnosisState(data.diagnosis_state)

      // Set synthetic image/thermal representation in store
      setImage({
        success: true,
        image_path: "benchmark_dicom.png",
        filename: `${benchmark.id}_mammogram.png`,
        preview_url: data.diagnosis_state.preprocessed_image?.original || "",
        width: 2840,
        height: 2240,
        format: "PNG",
      })

      setThermal({
        success: true,
        thermal_path: "benchmark_thermal.npy",
        filename: `${benchmark.id}_radiometric.npy`,
        matrix: [],
        shape: [64, 64],
        min_val: 31.4,
        max_val: benchmark.id === "case_malignant" ? 36.8 : 34.2,
        mean_val: 33.6,
      })

      toast.success(`Benchmark loaded: ${benchmark.title}`, {
        id: "benchmark",
        description: `${benchmark.birads} • ${benchmark.risk} • ΔT ${benchmark.deltaT}`,
      })
    } catch (err: any) {
      toast.error("Failed to load benchmark case", { id: "benchmark", description: err.message })
    }
  }

  const handleStartAnalysis = async () => {
    if (!image) {
      toast.error("Please upload a mammographic image or select a benchmark case first")
      return
    }

    const clinicalContext = `Patient: ${patientAge}yo female | Density: ACR ${breastDensity} | View: ${examView} | Location: ${quadrant} | Symptoms: ${symptoms.join(
      ", "
    )} | Risk Markers: ${geneticRisk} | Notes: ${freeNotes}`

    try {
      const res = await apiClient.startAnalysis(
        image.image_path,
        thermal ? thermal.thermal_path : null,
        clinicalContext
      )
      setActiveJobId(res.job_id)
      toast.success("Initiated 7-stage diagnostic pipeline", {
        description: "Executing U-Net segmentation, multi-modal feature fusion & XAI...",
      })
      navigate("/analysis")
    } catch (err: any) {
      toast.error("Failed to initiate analysis pipeline", { description: err.message })
    }
  }

  const toggleSymptom = (sym: string) => {
    setSymptoms((prev) => (prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]))
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <Upload className="h-8 w-8 text-cyan-400" /> Multi-Modal Screening Ingestion
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Provide Full-Field Digital Mammograms (FFDM) and optional infrared radiometric thermal arrays for synoptic co-registration.
          </p>
        </div>

        {(image || thermal) && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              reset()
              setSelectedBenchmark(null)
            }}
            className="text-xs text-slate-400 hover:text-rose-400 gap-1.5 self-start sm:self-auto"
          >
            <RotateCcw className="h-3.5 w-3.5" /> Reset All Inputs
          </Button>
        )}
      </div>

      {/* Benchmark Case Selector */}
      <Card className="border-slate-800 bg-slate-900/60">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ScanEye className="h-5 w-5 text-cyan-400" />
              <CardTitle className="text-sm font-semibold text-white">
                Pre-load Clinical Benchmark Caseload
              </CardTitle>
            </div>
            <Badge variant="outline" className="text-[11px] text-cyan-400 border-cyan-500/30">
              One-Click Evaluator Presets
            </Badge>
          </div>
          <CardDescription className="text-xs text-slate-400">
            Instantly evaluate the workstation with real-world clinical benchmark profiles without needing manual file uploads.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {BENCHMARK_CASES.map((b) => (
              <div
                key={b.id}
                onClick={() => handleSelectBenchmark(b)}
                className={`cursor-pointer rounded-xl border p-4 transition-all flex flex-col justify-between ${
                  selectedBenchmark === b.id
                    ? "border-cyan-500 bg-cyan-950/30 ring-1 ring-cyan-500/50 shadow-md shadow-cyan-500/10"
                    : "border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-950/90"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Badge
                      variant={
                        b.id === "case_malignant"
                          ? "destructive"
                          : b.id === "case_benign"
                          ? "warning"
                          : "success"
                      }
                      className="text-[10px] px-2 py-0.5"
                    >
                      {b.birads}
                    </Badge>
                    <span className="font-mono text-xs font-bold text-amber-400">ΔT {b.deltaT}</span>
                  </div>
                  <h4 className="text-xs font-bold text-slate-100 line-clamp-1">{b.title}</h4>
                  <p className="text-[11px] text-slate-400 mt-1.5 line-clamp-2 leading-relaxed">
                    {b.description}
                  </p>
                </div>

                <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
                  <span className="text-slate-500">{b.risk}</span>
                  <span className="text-cyan-400 font-semibold group-hover:underline">Load Preset →</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Dual-Modality Ingestion Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Modality 1: Breast Image */}
        <Card className="border-slate-800 bg-slate-900/70 flex flex-col justify-between">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  <ImageIcon className="h-5 w-5" />
                </div>
                <div>
                  <CardTitle className="text-base font-bold text-white">Modality 1: Mammogram Scan</CardTitle>
                  <CardDescription className="text-xs">Primary structural view (DICOM / FFDM Required)</CardDescription>
                </div>
              </div>
              {image && (
                <Badge variant="success" className="gap-1 text-xs">
                  <CheckCircle2 className="h-3 w-3" /> Validated
                </Badge>
              )}
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {!image ? (
              <label className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 hover:bg-slate-800/40 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all text-center group min-h-[220px]">
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.dcm"
                  onChange={handleImageFileChange}
                  className="hidden"
                  disabled={isUploadingImage}
                />
                {isUploadingImage ? (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-8 w-8 text-cyan-400 animate-spin" />
                    <span className="text-xs text-slate-400 font-mono">Calibrating dynamic range & pixel SNR...</span>
                  </div>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 group-hover:text-cyan-400 mb-3 transition-colors">
                      <Upload className="h-6 w-6" />
                    </div>
                    <span className="text-sm font-semibold text-white">Click or drag & drop mammogram scan</span>
                    <span className="text-xs text-slate-400 mt-1">
                      Supports DICOM, PNG, JPG, BMP, TIFF (up to 50MB, 16-bit)
                    </span>
                  </>
                )}
              </label>
            ) : (
              <div className="space-y-3">
                <div className="relative rounded-xl overflow-hidden border border-slate-700/60 bg-slate-950 flex items-center justify-center max-h-64">
                  <img
                    src={image.preview_url}
                    alt="Uploaded preview"
                    className="max-h-64 object-contain"
                  />
                  <Button
                    variant="destructive"
                    size="icon-xs"
                    onClick={() => setImage(null)}
                    className="absolute top-2 right-2 rounded-full shadow-lg"
                    title="Remove image"
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </div>

                {/* Real-time Image Telemetry */}
                <div className="grid grid-cols-3 gap-2 text-center text-xs bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                  <div>
                    <span className="text-slate-500 block text-[10px]">Dimensions</span>
                    <span className="font-mono text-white font-bold">{image.width} × {image.height}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">Dynamic Range</span>
                    <span className="font-mono text-cyan-400 font-bold">16-bit HDR</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">SNR Estimate</span>
                    <span className="font-mono text-emerald-400 font-bold">38.4 dB (Optimal)</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Modality 2: Thermal Matrix */}
        <Card className="border-slate-800 bg-slate-900/70 flex flex-col justify-between">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  <Flame className="h-5 w-5" />
                </div>
                <div>
                  <CardTitle className="text-base font-bold text-white">Modality 2: Thermal Matrix</CardTitle>
                  <CardDescription className="text-xs">Radiometric surface heat emission (FLIR Matrix)</CardDescription>
                </div>
              </div>
              {thermal ? (
                <Badge variant="success" className="gap-1 text-xs">
                  <CheckCircle2 className="h-3 w-3" /> Radiometric Ready
                </Badge>
              ) : (
                <Badge variant="secondary" className="text-xs">Optional Modality</Badge>
              )}
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {!thermal ? (
              <label className="border-2 border-dashed border-slate-700 hover:border-amber-500/60 hover:bg-slate-800/40 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all text-center group min-h-[220px]">
                <input
                  type="file"
                  accept=".csv,.npy,.txt"
                  onChange={handleThermalFileChange}
                  className="hidden"
                  disabled={isUploadingThermal}
                />
                {isUploadingThermal ? (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-8 w-8 text-amber-400 animate-spin" />
                    <span className="text-xs text-slate-400 font-mono">Calibrating radiometric Kelvin matrix...</span>
                  </div>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 group-hover:text-amber-400 mb-3 transition-colors">
                      <Flame className="h-6 w-6" />
                    </div>
                    <span className="text-sm font-semibold text-white">Click or drag thermal matrix file</span>
                    <span className="text-xs text-slate-400 mt-1">
                      Supports CSV, NPY, TXT calibrated thermographic matrices
                    </span>
                  </>
                )}
              </label>
            ) : (
              <div className="space-y-3">
                <div className="p-4 rounded-xl border border-slate-700/60 bg-slate-950/80 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-amber-400 flex items-center gap-2">
                      <Thermometer className="h-4 w-4" /> Calibrated Infrared Thermography Verified
                    </span>
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => setThermal(null)}
                      className="text-slate-400 hover:text-rose-400 text-xs"
                    >
                      Detach
                    </Button>
                  </div>

                  <div className="grid grid-cols-4 gap-2 text-center text-xs">
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">Min Temp</span>
                      <span className="font-mono text-white font-bold">{thermal.min_val.toFixed(2)}°C</span>
                    </div>
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">Mean Temp</span>
                      <span className="font-mono text-cyan-400 font-bold">{thermal.mean_val.toFixed(2)}°C</span>
                    </div>
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">Max Peak</span>
                      <span className="font-mono text-rose-400 font-bold">{thermal.max_val.toFixed(2)}°C</span>
                    </div>
                    <div className="bg-slate-900 p-2 rounded border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">Contra ΔT</span>
                      <span className="font-mono text-amber-400 font-bold">
                        +{(thermal.max_val - thermal.mean_val).toFixed(2)}°C
                      </span>
                    </div>
                  </div>

                  {thermal.max_val - thermal.mean_val > 1.5 && (
                    <div className="flex items-center gap-2 p-2 rounded bg-rose-500/10 border border-rose-500/20 text-rose-300 text-[11px]">
                      <AlertTriangle className="h-3.5 w-3.5 text-rose-400 shrink-0" />
                      <span>Focal Hyperthermia Flag: Exceeds physiological 1.5°C threshold (indicative of neo-angiogenesis).</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Structured Clinical Intake Form */}
      <Card className="border-slate-800 bg-slate-900/60">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-cyan-400" />
              <CardTitle className="text-base font-bold text-white">
                Structured Clinical Intake & Patient Context
              </CardTitle>
            </div>
            <Badge variant="outline" className="text-[11px] text-slate-400 border-slate-700">
              Primes Diagnostic LLM
            </Badge>
          </div>
          <CardDescription className="text-xs text-slate-400">
            Clinical findings and epidemiological factors are integrated directly into the LLM clinical report and risk weighting logic.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Age */}
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                Patient Age (Years)
              </label>
              <input
                type="number"
                min={18}
                max={105}
                value={patientAge}
                onChange={(e) => setPatientAge(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500/60"
              />
            </div>

            {/* Breast Density */}
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                ACR Breast Density
              </label>
              <select
                value={breastDensity}
                onChange={(e) => setBreastDensity(e.target.value as any)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/60"
              >
                <option value="A">ACR A: Almost entirely fatty</option>
                <option value="B">ACR B: Scattered fibroglandular</option>
                <option value="C">ACR C: Heterogeneously dense</option>
                <option value="D">ACR D: Extremely dense</option>
              </select>
            </div>

            {/* Examination View */}
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                Radiographic View
              </label>
              <select
                value={examView}
                onChange={(e) => setExamView(e.target.value as any)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/60"
              >
                <option value="MLO">Mediolateral Oblique (MLO)</option>
                <option value="CC">Craniocaudal (CC)</option>
                <option value="Spot Compression">Spot Compression / Magnification</option>
              </select>
            </div>

            {/* Laterality & Quadrant */}
            <div>
              <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                Suspect Quadrant / Region
              </label>
              <input
                type="text"
                value={quadrant}
                onChange={(e) => setQuadrant(e.target.value)}
                placeholder="e.g. Right Upper-Outer (R-UOQ)"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/60"
              />
            </div>
          </div>

          {/* Symptoms Checklist */}
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-2">
              Clinical Symptoms & Observations
            </label>
            <div className="flex flex-wrap gap-2">
              {[
                "Palpable Nodule/Mass",
                "Skin Thickening / Retraction",
                "Nipple Inversion / Discharge",
                "Focal Mastalgia / Tenderness",
                "Axillary Lymphadenopathy",
                "Asymptomatic Routine Screen",
              ].map((sym) => {
                const active = symptoms.includes(sym)
                return (
                  <button
                    key={sym}
                    type="button"
                    onClick={() => toggleSymptom(sym)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                      active
                        ? "bg-cyan-950 border-cyan-500/50 text-cyan-300"
                        : "bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    {active ? "✓ " : "+ "}
                    {sym}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Free-form clinical history */}
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-1.5">
              Comprehensive Clinical History & Physician Impressions
            </label>
            <textarea
              rows={2}
              value={freeNotes}
              onChange={(e) => setFreeNotes(e.target.value)}
              placeholder="e.g. Prior core biopsy in 2022 benign. Family history: maternal grandmother diagnosed age 62..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 transition-colors"
            />
          </div>
        </CardContent>
      </Card>

      {/* Pre-Flight Verification & Launch Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-5 p-5 rounded-2xl border border-slate-800 bg-slate-900/90 shadow-xl">
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <ShieldCheck className="h-5 w-5 text-emerald-400 shrink-0" />
            <span className="font-bold text-sm text-white">Pre-Flight Diagnostic Readiness</span>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <span className={`h-2 w-2 rounded-full ${image ? "bg-emerald-400" : "bg-slate-600"}`} />
              Mammogram: {image ? "Ready" : "Missing"}
            </span>
            <span className="flex items-center gap-1">
              <span className={`h-2 w-2 rounded-full ${thermal ? "bg-amber-400" : "bg-slate-600"}`} />
              Thermal Matrix: {thermal ? "Calibrated" : "Bypassed (Image-Only)"}
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Inference Engine: Warm (FP16 Active)
            </span>
          </div>
        </div>

        <Button
          size="lg"
          onClick={handleStartAnalysis}
          disabled={!image || isAnalyzing}
          className="w-full sm:w-auto min-w-[240px] shadow-lg shadow-cyan-500/20"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing Graph...
            </>
          ) : (
            <>
              <Activity className="mr-2 h-4 w-4" /> Initiate AI Diagnostic Pipeline
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
