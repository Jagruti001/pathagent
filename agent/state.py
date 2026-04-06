from typing import TypedDict, List


class ReportState(TypedDict):
    # Raw text extracted from PDF
    raw_text: str

    # Extracted lab values e.g. {"HbA1c": {"value": 6.2, "unit": "%", "reference_range": "4.0-5.6"}}
    lab_values: dict

    # Values flagged as abnormal e.g. ["HbA1c", "Fasting Glucose"]
    flagged_markers: List[str]

    # Retrieved medical guideline context from ChromaDB
    guideline_context: str

    # Disease risk clusters identified e.g. ["Pre-Diabetes", "Metabolic Syndrome"]
    risk_clusters: List[str]

    # Risk scores per disease e.g. {"Pre-Diabetes": {"score": "HIGH", "horizon": "12 months"}}
    risk_scores: dict

    # Final action plan as string
    action_plan: str

    # Full reasoning explanation for UI display
    reasoning: str
