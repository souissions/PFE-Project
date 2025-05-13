from ..LLMInterface import LLMInterface
from ..LLMEnums import GoogleEnums
import google.generativeai as genai
import logging
from typing import List, Union

class GoogleProvider(LLMInterface):
    def __init__(self, api_key: str, 
                 default_input_max_characters: int=1000,
                 default_generation_max_output_tokens: int=1000,
                 default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # Configure the Google API client
        genai.configure(api_key=self.api_key)
        self.client = genai
        
        self.enums = GoogleEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        
        if not self.generation_model_id:
            self.logger.error("Generation model for Google was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        # Prepare the model
        model = self.client.GenerativeModel(self.generation_model_id)
        
        # Convert chat history to Google's format if needed
        if chat_history:
            # Assuming chat_history is in OpenAI format, convert to Google's
            google_chat_history = []
            for msg in chat_history:
                google_chat_history.append({
                    'role': msg['role'],
                    'parts': [msg['content']]
                })
            chat = model.start_chat(history=google_chat_history)
            response = chat.send_message(prompt)
        else:
            response = model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_output_tokens,
                    'temperature': temperature
                }
            )

        if not response or not response.text:
            self.logger.error("Error while generating text with Google")
            return None

        return response.text

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        if not self.embedding_model_id:
            self.logger.error("Embedding model for Google was not set")
            return None
            
        if isinstance(text, str):
            text = [text]
            
        # Using the embeddings model
        model = self.client.GenerativeModel(self.embedding_model_id)
        
        try:
            embeddings = []
            for t in text:
                result = model.embed_content(t)
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            self.logger.error(f"Error while embedding text with Google: {str(e)}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }