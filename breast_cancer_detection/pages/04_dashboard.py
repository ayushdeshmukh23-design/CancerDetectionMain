from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from breast_cancer_detection.components.cards import metric_card
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state


def render():
    st.title("📈 Analytics Dashboard")
    state = st.session_state.get("diagnosis_state") or {}
    analysis_complete = bool(st.session_state.get("analysis_complete", False)) or bool(state.get("ensemble_result") is not None)
    if state and state.get("ensemble_result") is not None:
        st.session_state["last_complete_state"] = state
    if (not state) or (not analysis_complete) or (state.get("ensemble_result") is None):
        fallback = st.session_state.get("last_complete_state") or {}
        if fallback and fallback.get("ensemble_result") is not None:
            state = fallback
            st.warning("Showing last completed analytics (latest run still processing or incomplete).")
        else:
            demo = demo_diagnosis_state()
            if not analysis_complete:
                st.info("Showing seeded analytics until analysis completes.")
            else:
                st.warning("Analysis output incomplete; showing seeded analytics.")
            state = demo
    result = state.get("ensemble_result") or {}
    probs = result.get("probabilities", {"Normal": 0.33, "Benign": 0.33, "Malignant": 0.34})
    agreement = float(np.std(list(probs.values())))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Overall Confidence", f"{result.get('confidence', 0):.1%}")
    with c2:
        metric_card("Risk Score", f"{result.get('risk_score', 0):.1f}/100")
    with c3:
        metric_card("Reliability", f"{result.get('reliability', 0):.1%}")
    with c4:
        metric_card("Model Agreement", f"{(1-agreement):.1%}")

    row2a, row2b = st.columns(2)
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=[probs["Normal"], probs["Benign"], probs["Malignant"], probs["Normal"]],
            theta=["Normal", "Benign", "Malignant", "Normal"],
            fill="toself",
            name="Ensemble",
        )
    )
    radar.update_layout(template="plotly_dark", title="Confidence Radar")
    row2a.plotly_chart(radar, width="stretch")

    donut = go.Figure(go.Pie(labels=list(probs.keys()), values=list(probs.values()), hole=0.55))
    donut.update_layout(template="plotly_dark", title="Class Probability Donut")
    row2b.plotly_chart(donut, width="stretch")

    history = st.session_state.get("patient_history", [])
    if analysis_complete and result and result.get("risk_score") is not None:
        current_point = {
            "idx": len(history) + 1 if history else 1,
            "date": "Current",
            "risk_score": float(result.get("risk_score", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
            "reliability": float(result.get("reliability", 0.0)),
            "risk_level": result.get("risk_level", "Unknown"),
        }
        if not history or float(history[-1].get("risk_score", -1.0)) != current_point["risk_score"]:
            history = [*history, current_point]
            st.session_state["patient_history"] = history[-25:]
    if (not analysis_complete) and (not history):
        history = [{"idx": i, "date": f"Scan-{i}", "risk_score": max(0, min(100, result.get("risk_score", 50) + np.random.randn() * 5)), "confidence": result.get("confidence", 0.7), "reliability": result.get("reliability", 0.7)} for i in range(1, 26)]
    if analysis_complete and not history:
        history = [{
            "idx": 1,
            "date": "Current",
            "risk_score": float(result.get("risk_score", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
            "reliability": float(result.get("reliability", 0.0)),
            "risk_level": result.get("risk_level", "Unknown"),
        }]

    x = [h.get("idx", i + 1) for i, h in enumerate(history)]
    y = [h.get("risk_score", h.get("risk", 50.0)) for h in history]
    conf = [float(h.get("confidence", result.get("confidence", 0.7))) * 100 for h in history]
    rel = [float(h.get("reliability", result.get("reliability", 0.7))) * 100 for h in history]
    adv = make_subplots(rows=2, cols=2, subplot_titles=("Risk Progression", "Confidence vs Reliability", "Risk Distribution", "Rolling Trend"))
    adv.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Risk", line={"color": "#00D4FF"}), row=1, col=1)
    adv.add_trace(go.Scatter(x=conf, y=rel, mode="markers", name="Conf/Rel", marker={"color": "#7C3AED", "size": 9}), row=1, col=2)
    adv.add_trace(go.Histogram(x=y, nbinsx=12, marker_color="#10B981", name="Risk Dist"), row=2, col=1)
    rolling = pd.Series(y).rolling(5, min_periods=1).mean().tolist()
    adv.add_trace(go.Scatter(x=x, y=rolling, mode="lines", line={"color": "#F59E0B"}, name="Rolling Mean"), row=2, col=2)
    adv_title = "Advanced Analytics (Real-time Pipeline Data)" if analysis_complete else "Advanced Analytics (Seeded 25 Points)"
    adv.update_layout(template="plotly_dark", height=700, title=adv_title, showlegend=False)
    st.plotly_chart(adv, width="stretch")

    st.subheader("Analytics Table")
    if analysis_complete:
        st.caption("Displaying pipeline-derived case analytics.")
    else:
        st.caption("Displaying seeded demo analytics until pipeline completes.")
    st.dataframe(pd.DataFrame(history).head(25), width="stretch")


if __name__ == "__main__":
    render()

