from __future__ import annotations

import streamlit as st


def metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(text: str, level: str = "info"):
    st.markdown(f'<span class="status-badge status-{level}">{text}</span>', unsafe_allow_html=True)

