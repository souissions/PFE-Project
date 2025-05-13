#Graph Configuration & State Definition
# graph/Graph.py

# graph/graph.py

from langgraph.graph import StateGraph, END
from state import AgentState  # Import the AgentState class (you can keep this in state.py or move it here)

from tools import *  # To import all functions from tools.py
from utils import load_llm  # To load the LLM instance

MAX_REFINE_LOOPS = 2 # Max times to run the refine->evaluate loop


from GraphFlow import (
    classify_intent_node,
    gather_symptoms_node,
    check_triage_relevance_node,
    handle_info_request_node,
    handle_off_topic_node,
    handle_irrelevant_triage_node,
    perform_final_analysis_node,
    evaluate_explanation_node,
    refine_explanation_node,
    extract_specialist_and_doctors_node,
    prepare_final_output_node
)
from stores.langgraph.llm.templates.en.langgraphprompts import *  # Import prompts

# --- Initialize the LLM instance (make sure it's defined correctly in utils.py) ---
llm = load_llm()  # Initialize LLM here

# --- Define Route Functions (You can copy these from the previous answers) ---

def route_based_on_intent(state: AgentState) -> str:
    intent = state.get("user_intent", "OFF_TOPIC")
    if intent == "SYMPTOM_TRIAGE":
        return "gather_symptoms"
    elif intent == "MEDICAL_INFORMATION_REQUEST":
        return "handle_info_request"
    else:
        return "handle_off_topic"

def should_continue_symptom_gathering(state: AgentState) -> str:
    if state.get("final_response"):
        return END
    else:
        return "check_triage_relevance"

def route_based_on_relevance(state: AgentState) -> str:
    is_relevant = state.get("is_relevant", False)
    if is_relevant:
        return "perform_final_analysis"
    else:
        return "handle_irrelevant_triage"

def route_based_on_evaluation(state: AgentState) -> str:
    critique = state.get("evaluator_critique", "OK")
    loop = state.get("loop_count", 0)
    if critique.upper().startswith("REVISE") and loop < MAX_REFINE_LOOPS:
        return "refine_explanation"
    else:
        return "extract_specialist_and_doctors"

# --- Graph Construction ---
def build_graph():
    """
    Builds the LangGraph workflow and defines nodes and edges.
    This method organizes the flow of the conversation based on user intent.
    """
    if not llm:  # Ensure the LLM is loaded before building the graph
        print("Error: LLM not loaded. Unable to build graph.")
        return None
    
    print("Building LangGraph workflow...")
    workflow = StateGraph(AgentState)  # Initialize the state graph for managing the agent's state

    print("Adding nodes to the graph...")
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("gather_symptoms", gather_symptoms_node)
    workflow.add_node("check_triage_relevance", check_triage_relevance_node)
    workflow.add_node("handle_info_request", handle_info_request_node)
    workflow.add_node("handle_off_topic", handle_off_topic_node)
    workflow.add_node("handle_irrelevant_triage", handle_irrelevant_triage_node)
    workflow.add_node("perform_final_analysis", perform_final_analysis_node)
    workflow.add_node("evaluate_explanation", evaluate_explanation_node)
    workflow.add_node("refine_explanation", refine_explanation_node)
    workflow.add_node("extract_specialist_and_doctors", extract_specialist_and_doctors_node)
    workflow.add_node("prepare_final_output", prepare_final_output_node)

    # Set the starting point of the graph (entry point for the conversation)
    workflow.set_entry_point("classify_intent")

    print("Adding conditional edges between nodes...")
    # Define the conditions for the transitions between nodes
    workflow.add_conditional_edges("classify_intent", route_based_on_intent, {
        "gather_symptoms": "gather_symptoms",
        "handle_info_request": "handle_info_request",
        "handle_off_topic": "handle_off_topic"
    })

    workflow.add_conditional_edges("gather_symptoms", should_continue_symptom_gathering, {
        "check_triage_relevance": "check_triage_relevance",
        END: END  # End the flow if gathering symptoms is complete
    })

    workflow.add_conditional_edges("check_triage_relevance", route_based_on_relevance, {
        "perform_final_analysis": "perform_final_analysis",
        "handle_irrelevant_triage": "handle_irrelevant_triage"
    })

    workflow.add_edge("perform_final_analysis", "evaluate_explanation")

    workflow.add_conditional_edges("evaluate_explanation", route_based_on_evaluation, {
        "extract_specialist_and_doctors": "extract_specialist_and_doctors",
        "refine_explanation": "refine_explanation"
    })

    workflow.add_edge("refine_explanation", "evaluate_explanation")
    workflow.add_edge("extract_specialist_and_doctors", "prepare_final_output")
    workflow.add_edge("prepare_final_output", END)  # End the conversation after preparation

    # Adding final endpoints (if necessary)
    workflow.add_edge("handle_info_request", END)
    workflow.add_edge("handle_off_topic", END)
    workflow.add_edge("handle_irrelevant_triage", END)

    print("Compiling the graph...")
    compiled_graph = workflow.compile()  # Compile the graph after all nodes and edges are defined
    print("LangGraph compiled successfully.")

    return compiled_graph  # Return the compiled graph


# Optional main block for testing graph build standalone (only useful when running independently)
if __name__ == '__main__':
    print("\nTesting graph builder standalone...")
    try:
        if not llm: print("LLM missing, build check might fail.")
        graph_app_test = build_graph()  # Test the graph build function
        if graph_app_test: print("\nStandalone build check successful!")
        else: print("\nStandalone build check resulted in None graph app.")
    except Exception as e:
        print(f"\nError during standalone graph build test: {e}")
        traceback.print_exc()  # Show error details if any
