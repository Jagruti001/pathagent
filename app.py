import streamlit as st
from dotenv import load_dotenv
from utils.pdf_parser import extract_text_from_pdf
from agent.graph import run_agent
import json
import requests

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PathAgent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 PathAgent")
st.subheader("Predictive Health Risk Agent — *Your lab report knows your future*")
st.markdown("Upload your blood test report and PathAgent will analyze **patterns across all markers** to predict future health risks — powered entirely by **free, local AI** (Llama 3.1 + sentence-transformers).")

st.divider()

# ─────────────────────────────────────────────
# Check Ollama is running
# ─────────────────────────────────────────────
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

if not check_ollama():
    st.error("""
⚠️ **Ollama is not running!**

Please start Ollama before using PathAgent:
```
ollama serve
```
Then make sure Llama 3.1 is downloaded:
```
ollama pull llama3.1
```
""")
    st.stop()
else:
    st.success("✅ Ollama is running — Llama 3.1 ready")

# ─────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📄 Upload your Lab Report (PDF)",
    type=["pdf"],
    help="Upload a digital PDF lab report. Scanned images may not work."
)

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    if st.button("🚀 Analyze Report", type="primary", use_container_width=True):

        # ── Step 1: Extract PDF text ──
        with st.spinner("📄 Reading your lab report..."):
            raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text:
            st.error("Could not extract text. Please ensure it's a digital PDF, not a scanned image.")
            st.stop()

        # ── Step 2: Run Agent ──
        st.info("🤖 PathAgent is analyzing your report using local Llama 3.1... This may take 2-3 minutes.")

        progress = st.progress(0, text="Starting analysis...")

        status_steps = [
            (15, "🔍 Node 1: Extracting lab values..."),
            (35, "⚠️ Node 2: Detecting abnormal markers..."),
            (55, "🧠 Node 3: Analyzing marker combinations..."),
            (75, "📊 Node 4: Calculating risk scores..."),
            (90, "📋 Node 5: Generating action plan..."),
        ]

        import threading
        import time

        result_holder = {}
        error_holder = {}

        def run_in_background():
            try:
                result_holder["result"] = run_agent(raw_text)
            except Exception as e:
                error_holder["error"] = str(e)

        thread = threading.Thread(target=run_in_background)
        thread.start()

        step_idx = 0
        while thread.is_alive():
            if step_idx < len(status_steps):
                pct, msg = status_steps[step_idx]
                progress.progress(pct, text=msg)
                step_idx += 1
            time.sleep(15)  # local LLM is slower than API — longer wait per step

        thread.join()
        progress.progress(100, text="✅ Analysis complete!")

        if "error" in error_holder:
            st.error(f"Agent error: {error_holder['error']}")
            st.stop()

        result = result_holder["result"]
        st.divider()

        # ─────────────────────────────────────────────
        # Results
        # ─────────────────────────────────────────────

        # ── Risk Dashboard ──
        st.markdown("## 🎯 Risk Dashboard")
        risk_scores = result.get("risk_scores", {})

        if risk_scores:
            cols = st.columns(len(risk_scores))
            color_map = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

            for col, (disease, info) in zip(cols, risk_scores.items()):
                with col:
                    icon = color_map.get(info.get("score", "LOW"), "⚪")
                    st.metric(
                        label=f"{icon} {disease}",
                        value=info.get("score", "N/A"),
                        delta=f"Risk horizon: {info.get('horizon', 'N/A')}",
                        delta_color="inverse"
                    )
                    st.caption(info.get("evidence", ""))
        else:
            st.info("No significant risk clusters identified.")

        st.divider()

        # ── Action Plan ──
        st.markdown("## 📋 Action Plan & Summaries")
        st.markdown(result.get("action_plan", "No action plan generated."))

        st.divider()

        # ── Agent Reasoning ──
        with st.expander("🧠 View Agent Reasoning Chain"):
            st.markdown(result.get("reasoning", ""))

        # ── Raw Lab Values ──
        with st.expander("🔬 View Extracted Lab Values"):
            lab_values = result.get("lab_values", {})
            if lab_values:
                st.json(lab_values)
            else:
                st.info("No lab values extracted.")

        # ── Flagged Markers ──
        with st.expander("⚠️ View Flagged Abnormal Markers"):
            flagged = result.get("flagged_markers", [])
            if flagged:
                for marker in flagged:
                    st.markdown(f"- ⚠️ **{marker}**")
            else:
                st.success("All markers within normal range!")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("⚠️ PathAgent is a research prototype. It does NOT provide medical advice. Always consult a qualified physician.")
st.caption("🆓 Powered by Llama 3.1 (Ollama) + sentence-transformers/all-MiniLM-L6-v2 — 100% free, runs locally.")
