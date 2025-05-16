from ..LLMInterface import LLMInterface
import logging
from langchain_huggingface import HuggingFaceEmbeddings

class HuggingFaceProvider(LLMInterface):
    def __init__(self, model_name: str, default_input_max_characters: int=1000):
        self.model_name = model_name
        self.default_input_max_characters = default_input_max_characters
        self.embedding_model_id = model_name
        self.embedding_size = 384  # This is typical for MiniLM, adjust if needed
        self.logger = logging.getLogger(__name__)
        self.client = HuggingFaceEmbeddings(model_name=self.model_name)

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.client = HuggingFaceEmbeddings(model_name=model_id)

    def embed_text(self, text, document_type=None):
        if isinstance(text, str):
            text = [text]
        return self.client.embed_documents(text)

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": prompt}

    def set_generation_model(self, model_id: str):
        # Not used for embeddings, but required by interface
        pass

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None, temperature: float = None):
        # Not supported for HuggingFace embeddings-only provider
        raise NotImplementedError("Text generation is not supported for HuggingFaceProvider.")
