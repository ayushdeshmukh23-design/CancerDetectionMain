from __future__ import annotations

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from breast_cancer_detection.agents.diagnosis_agent import diagnosis_node
from breast_cancer_detection.agents.explainability_agent import explainability_node
from breast_cancer_detection.agents.feature_extractor_agent import feature_extractor_node
from breast_cancer_detection.agents.image_analyzer_agent import image_analyzer_node
from breast_cancer_detection.agents.report_generator_agent import report_generator_node
from breast_cancer_detection.agents.thermal_analyzer_agent import thermal_analyzer_node


class DiagnosisState(TypedDict):
    image_path: str
    thermal_matrix_path: Optional[str]
    patient_context: Optional[str]
    preprocessed_image: Optional[object]
    segmentation_mask: Optional[object]
    image_features: Optional[dict]
    thermal_features: Optional[dict]
    deep_embeddings: Optional[object]
    model_predictions: Optional[dict]
    ensemble_result: Optional[dict]
    gradcam_heatmap: Optional[str]
    shap_values: Optional[dict]
    lime_explanation: Optional[object]
    markdown_report: Optional[str]
    risk_assessment: Optional[dict]
    chat_history: List[dict]
    llm_response: Optional[str]
    pipeline_logs: List[str]
    errors: List[str]
    processing_time: float


import threading

_COMPILED_GRAPH = None
_GRAPH_LOCK = threading.Lock()


def build_diagnosis_graph():
    graph = StateGraph(DiagnosisState)
    graph.add_node("image_analyzer", image_analyzer_node)
    graph.add_node("thermal_analyzer", thermal_analyzer_node)
    graph.add_node("feature_extractor", feature_extractor_node)
    graph.add_node("diagnosis_agent", diagnosis_node)
    graph.add_node("explainability_agent", explainability_node)
    graph.add_node("report_generator", report_generator_node)
    graph.set_entry_point("image_analyzer")
    graph.add_conditional_edges(
        "image_analyzer",
        lambda state: "thermal_analyzer" if state.get("thermal_matrix_path") else "feature_extractor",
        {"thermal_analyzer": "thermal_analyzer", "feature_extractor": "feature_extractor"},
    )
    graph.add_edge("thermal_analyzer", "feature_extractor")
    graph.add_edge("feature_extractor", "diagnosis_agent")
    graph.add_edge("diagnosis_agent", "explainability_agent")
    graph.add_edge("explainability_agent", "report_generator")
    graph.add_edge("report_generator", END)
    return graph.compile()


def get_compiled_diagnosis_graph():
    """Thread-safe singleton compiled graph to eliminate re-compilation latency."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        with _GRAPH_LOCK:
            if _COMPILED_GRAPH is None:
                _COMPILED_GRAPH = build_diagnosis_graph()
    return _COMPILED_GRAPH


