import base64
import logging
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage

logger = logging.getLogger("uvicorn")

class MultimodalUtils:
    """Utility class for handling multimodal input preparation."""
    
    @staticmethod
    def prepare_llm_input(
        text_input: str,
        image_bytes: Optional[bytes] = None,
        use_history: bool = True,
        include_image: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Prepare multimodal input for LLM.
        
        Args:
            text_input (str): The text input to include
            image_bytes (Optional[bytes]): Optional image data
            use_history (bool): Whether to include conversation history
            include_image (bool): Whether to include image in output
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of content parts for LLM input
        """
        logger.info("🖼️ Preparing multimodal input...")
        content_list = []
        
        # Add text part
        if text_input:
            content_list.append({"type": "text", "text": text_input})
            logger.debug(f"Added text input: '{text_input[:100]}...'")
        
        # Add image part if available and requested
        if image_bytes and include_image:
            try:
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                mime_type = "image/jpeg"  # Assume JPEG for now
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
                }
                content_list.append(image_part)
                logger.info("✅ Added image to input")
            except Exception as e:
                logger.error(f"❌ Error encoding image: {e}")
        
        if not content_list:
            logger.warning("⚠️ No content to send to LLM")
            return None
            
        return content_list

    @staticmethod
    def create_multimodal_message(content_list: List[Dict[str, Any]]) -> HumanMessage:
        """
        Create a multimodal message for LLM.
        
        Args:
            content_list (List[Dict[str, Any]]): List of content parts
            
        Returns:
            HumanMessage: Message ready for LLM
        """
        return HumanMessage(content=content_list)

    @staticmethod
    def format_conversation_history(history: List[Dict[str, Any]], max_length: int = 50) -> str:
        """
        Format conversation history for inclusion in prompts.
        
        Args:
            history (List[Dict[str, Any]]): List of conversation messages
            max_length (int): Maximum length of each message to include
            
        Returns:
            str: Formatted conversation history
        """
        if not history:
            return ""
            
        formatted_history = []
        for msg in history[:-1]:  # Exclude the latest message
            role = msg.get('role', '?')
            content = msg.get('content', '?')
            formatted_history.append(f"{role}: {content[:max_length]}...")
            
        return "\n".join(formatted_history) 