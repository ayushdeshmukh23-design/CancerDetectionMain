from __future__ import annotations

import hashlib
import time

import streamlit as st

from breast_cancer_detection.agents.chat_agent import stream_chat_response
from breast_cancer_detection.utils.background_jobs import get_job, start_job
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state
from breast_cancer_detection.utils.file_store import render_download_button, save_output_text
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("chat_page")


def render():
    st.title("💬 AI Assistant")
    state = st.session_state.get("diagnosis_state")
    if state is None:
        fallback = st.session_state.get("last_complete_state")
        if fallback:
            state = fallback
            st.info("Loaded last completed diagnosis for chat context.")
        else:
            demo = demo_diagnosis_state()
            st.session_state["diagnosis_state"] = demo
            st.session_state["patient_history"] = demo["seeded_history"]
            st.session_state["analysis_complete"] = True
            state = demo
            st.info("No active diagnosis found — loaded seeded demo context for chat.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_active_job_id" not in st.session_state:
        st.session_state["chat_active_job_id"] = None
    if "chat_last_poll" not in st.session_state:
        st.session_state["chat_last_poll"] = 0.0

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.caption("Professional clinical assistant with context-aware answers from current analysis.")
    st.markdown("##### Suggested questions")

    suggestions = [
        "What does malignant mean?",
        "How confident is the AI?",
        "What should I do next?",
        "Explain the Grad-CAM result",
        "Summarize findings for a non-technical patient",
        "What are the key risk drivers in this case?",
        "How should I interpret reliability vs confidence?",
        "What follow-up tests should be considered?",
        "Compare benign vs malignant indicators in this report",
        "Give me a concise clinician handoff summary",
    ]
    suggestion_cols = st.columns([1, 1, 1])
    for i, s in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        if col.button(s, key=f"chat-suggestion-{i}", use_container_width=True, type="secondary"):
            st.session_state["chat_prompt"] = s

    for m in st.session_state["chat_history"]:
        role = "🧑‍⚕️ You" if m["role"] == "user" else "🤖 OncoVision"
        avatar = "🧑" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(f"**{role}**")
            st.write(m["content"])

    prompt = st.chat_input("Ask anything about your result...")
    if not prompt and st.session_state.get("chat_prompt"):
        prompt = st.session_state.pop("chat_prompt")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        report_context = (state or {}).get("markdown_report", "")
        history_snapshot = list(st.session_state["chat_history"][:-1])

        def _collect_stream():
            # NOTE: background thread must not touch st.session_state directly.
            stream = stream_chat_response(prompt, report_context, history_snapshot)
            return "".join(list(stream))

        st.session_state["chat_active_job_id"] = start_job("chat", _collect_stream)

    active_job = get_job(st.session_state.get("chat_active_job_id"))
    if active_job:
        if active_job["status"] == "running":
            st.info("Assistant is preparing response... You can switch tabs and come back.")
            now = time.time()
            if now - float(st.session_state.get("chat_last_poll", 0.0)) > 0.8:
                st.session_state["chat_last_poll"] = now
                st.rerun()
        elif active_job["status"] == "done":
            content = active_job.get("result", "")
            if content:
                if not st.session_state["chat_history"] or st.session_state["chat_history"][-1].get("content") != content:
                    st.session_state["chat_history"].append({"role": "assistant", "content": content})
            st.session_state["chat_active_job_id"] = None
            st.rerun()
        elif active_job["status"] == "failed":
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"Chat task failed: {active_job.get('error', 'Unknown error')}"}
            )
            st.session_state["chat_active_job_id"] = None

    _, action_right = st.columns([7, 3], vertical_alignment="center")
    with action_right:
        st.markdown("<div class='chat-actions'>", unsafe_allow_html=True)
        b1, b2 = st.columns([1, 1])
        if b1.button("Clear chat", use_container_width=True, key="chat-clear"):
            st.session_state["chat_history"] = []
    transcript = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["chat_history"]])
    transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    chat_path = st.session_state.get("chat_txt_path")
    prev_hash = st.session_state.get("chat_transcript_hash")
    try:
        if transcript_hash != prev_hash or not chat_path:
            chat_path = save_output_text(transcript, "chat_history.txt")
            st.session_state["chat_txt_path"] = chat_path
            st.session_state["chat_transcript_hash"] = transcript_hash
    except Exception as exc:
        logger.exception("Failed to persist chat transcript")
        st.error(f"Failed to save chat transcript: {exc}")
        chat_path = st.session_state.get("chat_txt_path")
    with action_right:
        with b2:
            render_download_button(chat_path, "Download chat", "text/plain", "download-chat-txt")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    render()

