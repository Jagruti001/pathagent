from langgraph.graph import StateGraph, END
from agent.state import ReportState
from agent.nodes import (
    extraction_node,
    anomaly_detection_node,
    combination_reasoning_node,
    risk_scoring_node,
    action_plan_node
)


def build_graph():
    graph = StateGraph(ReportState)

    # Add all 5 nodes
    graph.add_node("extraction", extraction_node)
    graph.add_node("anomaly_detection", anomaly_detection_node)
    graph.add_node("combination_reasoning", combination_reasoning_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("action_plan", action_plan_node)

    # Define flow: each node leads to the next
    graph.set_entry_point("extraction")
    graph.add_edge("extraction", "anomaly_detection")
    graph.add_edge("anomaly_detection", "combination_reasoning")
    graph.add_edge("combination_reasoning", "risk_scoring")
    graph.add_edge("risk_scoring", "action_plan")
    graph.add_edge("action_plan", END)

    return graph.compile()


# Singleton — built once, reused across requests
agent_graph = build_graph()


def run_agent(raw_text: str) -> ReportState:
    """Run the full PathAgent pipeline on extracted PDF text."""
    initial_state: ReportState = {
        "raw_text": raw_text,
        "lab_values": {},
        "flagged_markers": [],
        "guideline_context": "",
        "risk_clusters": [],
        "risk_scores": {},
        "action_plan": "",
        "reasoning": ""
    }
    return agent_graph.invoke(initial_state)
