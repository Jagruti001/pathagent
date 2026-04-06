from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from agent.state import ReportState
from agent.tools import retrieve_guidelines
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# Free local LLM via Ollama — no API key, no cost
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


def safe_parse_json(text: str, fallback):
    """Safely parse JSON from LLM output, handling markdown fences."""
    try:
        clean = text.strip()
        # Strip markdown code fences if present
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return fallback


# ─────────────────────────────────────────────
# NODE 1: Extract lab values from raw PDF text
# ─────────────────────────────────────────────
def extraction_node(state: ReportState) -> ReportState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical data extraction expert.
Extract ALL lab test values from the given report text.
Return ONLY a valid JSON object. No explanation. No markdown. Just raw JSON.
Format:
{{
  "HbA1c": {{"value": 6.2, "unit": "%", "reference_range": "4.0-5.6"}},
  "Fasting Glucose": {{"value": 108, "unit": "mg/dL", "reference_range": "70-99"}}
}}
If reference range is not mentioned, use null."""),
        ("human", "Extract lab values from this report:\n\n{raw_text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"raw_text": state["raw_text"]})
    lab_values = safe_parse_json(response.content, {})

    return {**state, "lab_values": lab_values}


# ─────────────────────────────────────────────
# NODE 2: Flag abnormal values
# ─────────────────────────────────────────────
def anomaly_detection_node(state: ReportState) -> ReportState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a lab result reviewer.
Given lab values with their reference ranges, identify which markers are ABNORMAL.
A marker is abnormal if its value is outside the reference range.
Return ONLY a JSON array of abnormal marker names. No explanation. No markdown.
Example: ["HbA1c", "Fasting Glucose", "Triglycerides"]"""),
        ("human", "Identify abnormal markers from:\n\n{lab_values}")
    ])

    chain = prompt | llm
    response = chain.invoke({"lab_values": json.dumps(state["lab_values"], indent=2)})
    flagged = safe_parse_json(response.content, [])

    return {**state, "flagged_markers": flagged}


# ─────────────────────────────────────────────
# NODE 3: Combination reasoning — THE UNIQUE PART
# ─────────────────────────────────────────────
def combination_reasoning_node(state: ReportState) -> ReportState:
    # Build a search query from flagged markers
    query = f"Disease risk for combination of: {', '.join(state['flagged_markers'])}"
    guideline_context = retrieve_guidelines(query)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert physician.
You are given a list of abnormal lab markers and relevant medical guidelines.
Identify disease risk CLUSTERS — combinations of markers that together indicate a specific disease.
Do NOT analyze markers individually. Look at PATTERNS across multiple markers.
Return ONLY a JSON array of risk cluster names. No explanation. No markdown.
Example: ["Pre-Diabetes", "Metabolic Syndrome", "Hypothyroidism Risk"]"""),
        ("human", """Abnormal markers: {flagged_markers}

Relevant medical guidelines:
{guideline_context}

Return only a JSON array of disease risk clusters.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "flagged_markers": json.dumps(state["flagged_markers"]),
        "guideline_context": guideline_context
    })
    clusters = safe_parse_json(response.content, [])

    return {**state, "risk_clusters": clusters, "guideline_context": guideline_context}


# ─────────────────────────────────────────────
# NODE 4: Risk scoring with time horizon
# ─────────────────────────────────────────────
def risk_scoring_node(state: ReportState) -> ReportState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical risk assessment expert.
For each disease risk cluster, assign a risk score.
Return ONLY a valid JSON object. No explanation. No markdown.
Format:
{{
  "Pre-Diabetes": {{
    "score": "HIGH",
    "horizon": "12 months",
    "evidence": "HbA1c 6.4% and Fasting Glucose 112 mg/dL together match ADA pre-diabetic criteria"
  }}
}}
score must be one of: LOW, MEDIUM, HIGH"""),
        ("human", """Risk clusters: {risk_clusters}
Lab values: {lab_values}
Guidelines: {guideline_context}

Return only the JSON risk scores object.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "risk_clusters": json.dumps(state["risk_clusters"]),
        "lab_values": json.dumps(state["lab_values"], indent=2),
        "guideline_context": state["guideline_context"][:1000]  # limit context length for local LLM
    })
    risk_scores = safe_parse_json(response.content, {})

    return {**state, "risk_scores": risk_scores}


# ─────────────────────────────────────────────
# NODE 5: Generate action plan
# ─────────────────────────────────────────────
def action_plan_node(state: ReportState) -> ReportState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a preventive healthcare advisor.
Based on the patient's risk scores, generate a structured report.
Format your response exactly like this:

## 🧑 For You (Patient Summary)
[Write in plain English. No medical jargon. Explain what the risks mean in simple terms.]

## 👨‍⚕️ For Your Doctor (Clinical Summary)
[Write in clinical language with specific marker values and risk levels.]

## 📋 Recommended Actions
- [Action 1]
- [Action 2]
- [Action 3]

## 🔁 Follow-up Tests
[Which markers to retest and when]"""),
        ("human", """Risk scores: {risk_scores}
Lab values: {lab_values}

Generate the full report.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "risk_scores": json.dumps(state["risk_scores"], indent=2),
        "lab_values": json.dumps(state["lab_values"], indent=2)
    })

    reasoning = f"""**Flagged Markers:** {', '.join(state['flagged_markers'])}

**Identified Risk Clusters:** {', '.join(state['risk_clusters'])}

**Risk Scores:**
{json.dumps(state['risk_scores'], indent=2)}

**Guideline Context Used (first 500 chars):**
{state['guideline_context'][:500]}...
"""

    return {**state, "action_plan": response.content, "reasoning": reasoning}
