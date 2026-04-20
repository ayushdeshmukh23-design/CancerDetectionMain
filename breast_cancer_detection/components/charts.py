from __future__ import annotations

import plotly.graph_objects as go


def confidence_gauge(value: float, title: str = "Confidence"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100.0,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00D4FF"},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16,185,129,0.20)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.20)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.20)"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=300)
    return fig


def probability_bars(prob_map):
    labels = list(prob_map.keys())
    values = [v * 100 for v in prob_map.values()]
    colors = ["#10B981", "#F59E0B", "#EF4444"]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(template="plotly_dark", xaxis_title="Probability (%)", yaxis_title="")
    return fig

