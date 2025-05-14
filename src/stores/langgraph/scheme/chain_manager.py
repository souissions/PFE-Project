import logging
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger("uvicorn")

class ChainManager:
    """Manager class for LLM chains and their initialization."""
    
    def __init__(self, model_name: str = "gpt-4-vision-preview"):
        """
        Initialize the chain manager.
        
        Args:
            model_name (str): Name of the OpenAI model to use
        """
        self.model_name = model_name
        self.chains: Dict[str, Any] = {}
        self._initialize_chains()
        
    def _initialize_chains(self):
        """Initialize all required chains."""
        logger.info("🔧 Initializing LLM chains...")
        
        # Initialize the base model
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize chains
        self._init_intent_classifier_chain()
        self._init_symptom_gatherer_chain()
        self._init_relevance_checker_chain()
        self._init_final_analysis_chain()
        self._init_explanation_evaluator_chain()
        self._init_explanation_refiner_chain()
        self._init_off_topic_handler_chain()
        self._init_info_request_handler_chain()
        
        logger.info("✅ All chains initialized successfully")
        
    def _init_intent_classifier_chain(self):
        """Initialize the intent classification chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Classify the user's intent into one of: 'triage', 'info_request', or 'off_topic'."),
            ("human", "{input}")
        ])
        
        self.chains["intent_classifier"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_symptom_gatherer_chain(self):
        """Initialize the symptom gathering chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Extract and list all symptoms mentioned by the user."),
            ("human", "{input}")
        ])
        
        self.chains["symptom_gatherer"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_relevance_checker_chain(self):
        """Initialize the relevance checking chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Evaluate if the symptoms are relevant for medical triage."),
            ("human", "Symptoms: {symptoms}\n\nEvaluate relevance.")
        ])
        
        self.chains["relevance_checker"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_final_analysis_chain(self):
        """Initialize the final analysis chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Provide a final analysis of the patient's condition."),
            ("human", "Symptoms: {symptoms}\n\nProvide analysis.")
        ])
        
        self.chains["final_analysis"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_explanation_evaluator_chain(self):
        """Initialize the explanation evaluation chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Evaluate the quality of the explanation."),
            ("human", "Explanation: {explanation}\n\nEvaluate quality.")
        ])
        
        self.chains["explanation_evaluator"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_explanation_refiner_chain(self):
        """Initialize the explanation refinement chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Refine the explanation to be more clear and accurate."),
            ("human", "Original explanation: {explanation}\n\nRefine it.")
        ])
        
        self.chains["explanation_refiner"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_off_topic_handler_chain(self):
        """Initialize the off-topic handling chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Handle off-topic queries professionally."),
            ("human", "{input}")
        ])
        
        self.chains["off_topic_handler"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def _init_info_request_handler_chain(self):
        """Initialize the information request handling chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage assistant. Answer general medical information requests."),
            ("human", "{input}")
        ])
        
        self.chains["info_request_handler"] = (
            prompt | self.model | StrOutputParser()
        )
        
    def get_chain(self, chain_name: str) -> Optional[Any]:
        """
        Get a specific chain by name.
        
        Args:
            chain_name (str): Name of the chain to retrieve
            
        Returns:
            Optional[Any]: The requested chain or None if not found
        """
        chain = self.chains.get(chain_name)
        if not chain:
            logger.warning(f"⚠️ Chain '{chain_name}' not found")
        return chain
        
    def run_chain(self, chain_name: str, **kwargs) -> Optional[str]:
        """
        Run a specific chain with the given inputs.
        
        Args:
            chain_name (str): Name of the chain to run
            **kwargs: Input parameters for the chain
            
        Returns:
            Optional[str]: Chain output or None if chain not found
        """
        chain = self.get_chain(chain_name)
        if not chain:
            return None
            
        try:
            logger.info(f"🔄 Running chain: {chain_name}")
            result = chain.invoke(kwargs)
            logger.info(f"✅ Chain '{chain_name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Error running chain '{chain_name}': {e}")
            return None 