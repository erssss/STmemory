from typing import Literal, Optional

from stmem.integrations.embeddings.base import EmbeddingBase
from stmem.integrations.embeddings.config.base import BaseEmbedderConfig

try:
    from ollama import Client
except ImportError:
    Client = None


class OllamaEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "nomic-embed-text"
        self.config.embedding_dims = self.config.embedding_dims or 512

        self.ollama_base_url = getattr(self.config, "ollama_base_url", None) or "http://localhost:11434"

        self.client = None
        if Client is not None:
            self.client = Client(host=self.ollama_base_url)
            self._ensure_model_exists()

    def _ensure_model_exists(self):
        """
        Ensure the specified model exists locally. If not, pull it from Ollama.
        """
        if not self.client:
            return
        local_models = self.client.list()["models"]
        if not any(
            model.get("name") == self.config.model
            or model.get("model") == self.config.model
            for model in local_models
        ):
            self.client.pull(self.config.model)

    def embed(
        self,
        text,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ):
        """
        Get the embedding for the given text using Ollama.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use.
                Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        if self.client:
            response = self.client.embeddings(
                model=self.config.model,
                prompt=text,
            )
            return response["embedding"]

        import requests

        api_url = f"{str(self.ollama_base_url).rstrip('/')}/api/embeddings"
        resp = requests.post(
            api_url,
            json={"model": self.config.model, "prompt": text},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        return data["embedding"]
