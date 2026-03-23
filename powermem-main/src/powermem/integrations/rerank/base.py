"""
Base class for rerank models
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from powermem.integrations.rerank.config.base import BaseRerankConfig


class RerankBase(ABC):
    """Base class for rerank models
    
    Args:
        config (Optional[BaseRerankConfig]): Configuration for the rerank model
    """

    def __init__(self, config: Optional[BaseRerankConfig] = None):
        if config is None:
            self.config = BaseRerankConfig()
        else:
            self.config = config

    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query (str): The search query
            documents (List[str]): List of document texts to rerank
            top_n (Optional[int]): Number of top results to return. If None, uses config.top_n

        Returns:
            List[Tuple[int, float]]: List of (document_index, relevance_score) tuples,
                                     sorted by relevance score in descending order
        """
        pass

