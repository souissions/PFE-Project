#Graph Configuration & State Definition
# graph/Graph.py


from langgraph.graph import StateGraph, END
from stores.langgraph.scheme.state import AgentState
from stores.langgraph.graphFlow import GraphFlow
from typing import Callable, Dict, Any
import logging

logger = logging.getLogger("uvicorn")

class Graph:
    def __init__(self, graph_flow: GraphFlow):
        self.graph_flow = graph_flow
        self.graph = StateGraph(AgentState)
        self.MAX_REFINE_LOOPS = 2

    async def classify_intent(self, state: AgentState):
        return await self.graph_flow.classify_intent(state)

    async def check_relevance(self, state: AgentState):
        """Public method to check triage relevance using GraphFlow."""
        return await self.graph_flow.check_triage_relevance(state)

    # Define conditional edge handlers that correctly return the routing decision
    async def route_based_on_intent(self, state: AgentState) -> str:
        """Routes based on user intent classification."""
        logger.info("🔄 Routing based on intent...")
        new_state = await self.graph_flow.classify_intent(state)
        intent = new_state.get("intent", "OFF_TOPIC")  # <-- Use correct key
        logger.info(f"➡️ Routed to: {intent}")
        
        if intent == "SYMPTOM_TRIAGE":
            return "gather_symptoms"
        elif intent == "MEDICAL_INFORMATION_REQUEST":
            return "handle_info_request"
        else:
            return "handle_off_topic"

    async def should_continue_symptom_gathering(self, state: AgentState) -> str:
        """Decides whether to continue gathering symptoms or proceed to relevance check."""
        logger.info("🔄 Checking symptom gathering status...")
        new_state = await self.graph_flow.gather_symptoms(state)
        # If the system needs more symptom detail, loop back to gather_symptoms for multi-turn followup
        if new_state.get("needs_more_symptom_detail"):
            logger.info("🛑 Insufficient symptom detail, requesting more from user. Looping back to gather_symptoms.")
            return "gather_symptoms"
        # If a final response was set (e.g., off-topic or error), end
        if new_state.get("final_response"):
            logger.info("✅ Symptom gathering complete")
            return END
        # Otherwise, proceed to triage relevance check
        logger.info("➡️ Proceeding to relevance check")
        return "check_triage_relevance"

    async def route_based_on_relevance(self, state: AgentState) -> str:
        """Routes based on symptom relevance."""
        logger.info("🔄 Routing based on relevance...")
        new_state = await self.graph_flow.check_triage_relevance(state)
        is_relevant = new_state.get("is_relevant", False)
        logger.info(f"➡️ Relevance: {is_relevant}")
        
        if is_relevant:
            return "perform_final_analysis"
        else:
            return "handle_irrelevant_triage"

    async def route_based_on_evaluation(self, state: AgentState) -> str:
        """Routes based on explanation evaluation."""
        logger.info("🔄 Routing based on evaluation...")
        new_state = await self.graph_flow.evaluate_explanation(state)
        critique = new_state.get("evaluator_critique", "OK")
        loop = state.get("loop_count", 0)
        logger.info(f"➡️ Evaluation: {critique}, Loop: {loop}")
        
        if critique.upper().startswith("REVISE") and loop < self.MAX_REFINE_LOOPS:
            return "refine_explanation"
        else:
            return "prepare_final_output"

    async def prepare_final_output(self, state: AgentState):
        """Public method to prepare the final output using GraphFlow."""
        return await self.graph_flow.prepare_final_output(state)

    async def run_full_triage(self, state: AgentState):
        """
        Runs the full LangGraph triage pipeline from entry to end, returning the final state.
        This builds and executes the compiled state machine graph with the provided state.
        """
        compiled_graph = self.build()  # Get the compiled graph (StateGraph)
        # The compiled graph is an async runnable: use .ainvoke(state)
        result = await compiled_graph.ainvoke(state)
        return result

    def build(self):
        """Builds and compiles the graph with all nodes and edges."""
        logger.info("🏗️ Building graph...")
        
        # Add all nodes
        self.graph.add_node("classify_intent", self.graph_flow.classify_intent)
        self.graph.add_node("gather_symptoms", self.graph_flow.gather_symptoms)
        self.graph.add_node("check_triage_relevance", self.graph_flow.check_triage_relevance)
        self.graph.add_node("handle_info_request", self.graph_flow.handle_info_request)
        self.graph.add_node("handle_off_topic", self.graph_flow.handle_off_topic)
        self.graph.add_node("handle_irrelevant_triage", self.graph_flow.handle_irrelevant_triage)
        self.graph.add_node("perform_final_analysis", self.graph_flow.perform_final_analysis)
        self.graph.add_node("evaluate_explanation", self.graph_flow.evaluate_explanation)
        self.graph.add_node("refine_explanation", self.graph_flow.refine_explanation)
        self.graph.add_node("prepare_final_output", self.graph_flow.prepare_final_output)

        # Set entry point
        self.graph.set_entry_point("classify_intent")

        # Add conditional edges
        logger.info("🔗 Adding conditional edges...")
        
        # Intent-based routing
        self.graph.add_conditional_edges(
            "classify_intent",
            self.route_based_on_intent,
            {
                "gather_symptoms": "gather_symptoms",
                "handle_info_request": "handle_info_request",
                "handle_off_topic": "handle_off_topic"
            }
        )

        # Symptom gathering flow
        self.graph.add_conditional_edges(
            "gather_symptoms",
            self.should_continue_symptom_gathering,
            {
                "gather_symptoms": "gather_symptoms",  # <-- Fix: allow looping for follow-up
                "check_triage_relevance": "check_triage_relevance",
                END: END
            }
        )

        # Relevance-based routing
        self.graph.add_conditional_edges(
            "check_triage_relevance",
            self.route_based_on_relevance,
            {
                "perform_final_analysis": "perform_final_analysis",
                "handle_irrelevant_triage": "handle_irrelevant_triage"
            }
        )

        # Analysis and evaluation flow
        self.graph.add_edge("perform_final_analysis", "evaluate_explanation")
        
        # Evaluation-based routing
        self.graph.add_conditional_edges(
            "evaluate_explanation",
            self.route_based_on_evaluation,
            {
                "refine_explanation": "refine_explanation",
                "prepare_final_output": "prepare_final_output"
            }
        )

        # Refinement loop
        self.graph.add_edge("refine_explanation", "evaluate_explanation")
        
        # Final output preparation
        self.graph.add_edge("prepare_final_output", END)

        # Add direct end points
        self.graph.add_edge("handle_info_request", END)
        self.graph.add_edge("handle_off_topic", END)
        self.graph.add_edge("handle_irrelevant_triage", END)

        logger.info("✅ Graph built successfully")
        logger.info(f"Registered graph nodes: {list(self.graph.nodes.keys())}")
        return self.graph.compile()

# Optional main block for testing
if __name__ == '__main__':
    logger.info("🧪 Testing graph builder...")
    try:
        # TODO: Pass a real nlp_service instance here for full functionality
        raise RuntimeError("You must provide an nlp_service instance to GraphFlow!")
        # graph_flow = GraphFlow(AgentState(), nlp_service=your_nlp_service_instance)
        # graph = Graph(graph_flow)
        # compiled_graph = graph.build()
        # if compiled_graph:
        #     logger.info("✅ Graph compilation successful!")
        # else:
        #     logger.error("❌ Graph compilation failed")
    except Exception as e:
        logger.error(f"❌ Error during graph build test: {e}")
        import traceback
        traceback.print_exc()

# NOTE: You must pass nlp_service to GraphFlow in production code!
# The application (e.g., Streamlit app) is responsible for creating:
#   graph_flow = GraphFlow(AgentState(), nlp_service=your_nlp_service_instance)
#   graph = Graph(graph_flow)
#   compiled_graph = graph.build()
# Do NOT create global graph_flow/graph/compiled_graph here.

# Export the Graph class for use by the application
__all__ = ['Graph']
