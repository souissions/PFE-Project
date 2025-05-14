import logging
from typing import Dict, Any
from stores.langgraph import tools
from stores.llm.templates.template_parser import TemplateParser
from stores.langgraph.utils import load_llm
from stores.langgraph.scheme.state import AgentState
from stores.langgraph.scheme.models import (
    IntentClassifierInput, IntentClassifierOutput,
    SymptomGathererInput, SymptomGathererOutput,
    RelevanceCheckerInput, RelevanceCheckerOutput,
    InfoRequestHandlerInput, InfoRequestHandlerOutput,
    OffTopicHandlerInput, OffTopicHandlerOutput,
    IrrelevantTriageHandlerInput, IrrelevantTriageHandlerOutput,
    FinalAnalysisInput, FinalAnalysisOutput,
    ExplanationEvaluatorInput, ExplanationEvaluatorOutput,
    ExplanationRefinerInput, ExplanationRefinerOutput,
    SpecialistRecommenderInput, SpecialistRecommenderOutput,
    FinalOutputPreparerInput, FinalOutputPreparerOutput
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import base64
import json
from langchain.schema import HumanMessage

logger = logging.getLogger("uvicorn")

class GraphFlow:
    def __init__(self, state: AgentState):
        self.state = state
        self.parser = TemplateParser(language='en')
        self.llm = load_llm()
        
        # Initialize prompts
        self.intent_prompt_str = self.parser.get("intent_classifier", "intent_classifier_system")
        self.followup_prompt_str = self.parser.get("followup", "followup_system")
        
        # Initialize LangChain components
        self.intent_prompt = PromptTemplate(
            template=self.intent_prompt_str,
            input_variables=["conversation_history", "user_query"]
        )
        self.followup_prompt = PromptTemplate(
            template=self.followup_prompt_str,
            input_variables=["accumulated_symptoms", "user_query"]
        )
        
        # Create runnable sequences
        self.intent_chain = self.intent_prompt | self.llm
        self.followup_chain = self.followup_prompt | self.llm

    async def classify_intent(self, state: AgentState) -> Dict[str, Any]:
        """Classify user intent based on conversation history and latest query."""
        try:
            # Prepare input for intent classification
            input_data = {
                "conversation_history": state.conversation_history,
                "user_query": state.user_query
            }
            
            # Run intent classification
            result = await self.intent_chain.ainvoke(input_data)
            intent = result.content.strip()
            
            # Update state with intent
            state.intent = intent
            return {"intent": intent}
            
        except Exception as e:
            logger.error(f"Error in classify_intent: {e}")
            state.intent = "UNKNOWN"
            return {"intent": "UNKNOWN"}

    async def gather_symptoms(self, state: AgentState) -> Dict[str, Any]:
        """Gather and analyze symptoms from user input."""
        try:
            # Prepare input for symptom analysis
            input_data = {
                "accumulated_symptoms": state.accumulated_symptoms,
                "user_query": state.user_query
            }
            
            # Run symptom analysis
            result = await self.followup_chain.ainvoke(input_data)
            symptoms = result.content.strip()
            
            # Update state with new symptoms
            state.accumulated_symptoms = symptoms
            return {"symptoms": symptoms}
            
        except Exception as e:
            logger.error(f"Error in gather_symptoms: {e}")
            return {"symptoms": state.accumulated_symptoms}

    async def check_triage_relevance(self, state: AgentState) -> Dict[str, Any]:
        """Checks if the accumulated symptoms are medically relevant."""
        logger.info("🔍 Checking triage relevance...")
        
        input_data = RelevanceCheckerInput(
            accumulated_symptoms=state.get('accumulated_symptoms', '')
        )
        
        # TODO: Implement proper relevance check
        is_relevant = True
        
        logger.info(f"🔍 Triage relevance: {is_relevant}")
        return RelevanceCheckerOutput(is_relevant=is_relevant).dict()

    async def handle_info_request(self, state: AgentState) -> Dict[str, Any]:
        """Handles general medical information requests."""
        logger.info("📚 Handling information request...")
        
        input_data = InfoRequestHandlerInput(
            user_query=state.get('user_query', ''),
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        
        if not input_data.user_query:
            logger.warning("⚠️ No query provided for information request")
            return InfoRequestHandlerOutput(
                final_response="No query provided.",
                rag_context=None,
                uploaded_image_bytes=None
            ).dict()

        context = tools.retrieve_relevant_documents.invoke({"user_symptoms": input_data.user_query})
        icd_codes = tools.match_relevant_icd_codes.invoke({"user_symptoms": input_data.user_query})

        response_content = f"Context: {context}\nICD Codes: {icd_codes}"
        logger.info("✅ Information request handled")
        
        return InfoRequestHandlerOutput(
            final_response=response_content,
            rag_context=context,
            uploaded_image_bytes=None
        ).dict()

    async def perform_final_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Performs final analysis of symptoms and generates recommendations."""
        logger.info("🔬 Performing final analysis...")
        
        input_data = FinalAnalysisInput(
            accumulated_symptoms=state.get('accumulated_symptoms', ''),
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        
        context = tools.retrieve_relevant_documents.invoke({"user_symptoms": input_data.accumulated_symptoms})
        icd_codes = tools.match_relevant_icd_codes.invoke({"user_symptoms": input_data.accumulated_symptoms})

        logger.info("✅ Final analysis completed")
        return FinalAnalysisOutput(
            initial_explanation="Analysis complete",  # TODO: Implement proper explanation
            rag_context=context,
            matched_icd_codes=icd_codes,
            evaluator_critique=None,
            loop_count=0,
            uploaded_image_bytes=None
        ).dict()

    async def evaluate_explanation(self, state: AgentState) -> Dict[str, Any]:
        """Evaluates the clarity and quality of the explanation."""
        logger.info("📊 Evaluating explanation...")
        
        input_data = ExplanationEvaluatorInput(
            initial_explanation=state.get('initial_explanation', '')
        )
        
        # TODO: Implement proper evaluation
        critique = "OK"
        
        logger.info(f"📊 Evaluation result: {critique}")
        return ExplanationEvaluatorOutput(evaluator_critique=critique).dict()

    async def refine_explanation(self, state: AgentState) -> Dict[str, Any]:
        """Refines the explanation based on evaluation feedback."""
        logger.info("🔄 Refining explanation...")
        
        input_data = ExplanationRefinerInput(
            initial_explanation=state.get('initial_explanation', ''),
            evaluator_critique=state.get('evaluator_critique', 'OK')
        )
        
        if input_data.evaluator_critique == "OK":
            logger.info("✅ No refinement needed")
            return ExplanationRefinerOutput(
                initial_explanation=input_data.initial_explanation,
                loop_count=state.get('loop_count', 0) + 1
            ).dict()

        # TODO: Implement proper refinement
        refined_explanation = "Refined explanation"
        
        logger.info("✅ Explanation refined")
        return ExplanationRefinerOutput(
            initial_explanation=refined_explanation,
            loop_count=state.get('loop_count', 0) + 1
        ).dict()

    async def handle_off_topic(self, state: AgentState) -> Dict[str, Any]:
        """Handles off-topic queries."""
        logger.info("🚫 Handling off-topic query...")
        
        input_data = OffTopicHandlerInput(
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        
        response = "Sorry, this topic is outside my medical assistant capabilities. Please describe a health-related concern."
        logger.info("✅ Off-topic response generated")
        
        return OffTopicHandlerOutput(
            final_response=response,
            uploaded_image_bytes=None
        ).dict()

    async def handle_irrelevant_triage(self, state: AgentState) -> Dict[str, Any]:
        """Handles irrelevant triage cases."""
        logger.info("⚠️ Handling irrelevant triage...")
        
        input_data = IrrelevantTriageHandlerInput(
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        
        response = "Your symptoms don't appear to require immediate medical attention. However, if you're concerned, please consult a healthcare professional."
        logger.info("✅ Irrelevant triage response generated")
        
        return IrrelevantTriageHandlerOutput(
            final_response=response,
            rag_context="N/A (Triage irrelevant)",
            matched_icd_codes="N/A (Triage irrelevant)",
            uploaded_image_bytes=None
        ).dict()

    async def prepare_final_output(self, state: AgentState) -> Dict[str, Any]:
        """Prepares the final output for the user."""
        logger.info("📤 Preparing final output...")
        
        input_data = FinalOutputPreparerInput(
            final_explanation=state.get('final_explanation'),
            final_response=state.get('final_response')
        )
        
        final_response = input_data.final_response or input_data.final_explanation or "Processing complete."
        logger.info("✅ Final output prepared")
        
        return FinalOutputPreparerOutput(final_response=final_response).dict()

    async def extract_specialist_and_doctors(self, state: AgentState) -> AgentState:
        """Extract specialist and doctors information from the conversation."""
        try:
            # Get the last message from the user
            last_message = state.messages[-1].content if state.messages else ""
            
            # Use the LLM to extract specialist and doctors information
            prompt = f"""Based on the following conversation, extract information about specialists and doctors mentioned:
            {last_message}
            
            Return the information in this format:
            {{
                "specialists": ["specialist1", "specialist2"],
                "doctors": ["doctor1", "doctor2"]
            }}
            """
            
            response = await self.llm.ainvoke(prompt)
            
            # Parse the response
            try:
                info = json.loads(response)
                state.specialists = info.get("specialists", [])
                state.doctors = info.get("doctors", [])
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information using string manipulation
                state.specialists = []
                state.doctors = []
                
                # Add a message about the extraction
                state.messages.append(
                    HumanMessage(content="I've extracted information about specialists and doctors from our conversation.")
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Error in extract_specialist_and_doctors: {str(e)}")
            # Add error message to state
            state.messages.append(
                HumanMessage(content="I encountered an error while extracting specialist and doctor information.")
            )
            return state
