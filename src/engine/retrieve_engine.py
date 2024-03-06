

"""Base query engine."""

import logging
from typing import Any, List, Optional

from src.callbacks.callback_manager import CallbackManager
from src.node.base_node import NodeWithScore
from src.retriever.types import QueryBundle, QueryType
from src.retriever.base_retriver import BaseRetriever
from src.reranker.base_reranker import BaseReranker

from .base import BaseEngine

logger = logging.getLogger(__name__)

class RetriverEngine(BaseEngine):
    """Base query engine."""

    def __init__(
        self, 
        callback_manager: Optional[CallbackManager],
        retriever: BaseRetriever,
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        super().__init__(
            callback_manager=callback_manager
        )

    def run_engine(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.search(query_bundle=query_bundle)
    
    async def arun_engine(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self.asearch(query_bundle=query_bundle)

    def search(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self.retriever.retrieve(query_bundle)
        nodes = self._rerank_nodes(
            nodes=nodes,
            query_bundle=query_bundle
        )
        return nodes

    async def asearch(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self.retriever.aretrieve(query_bundle)
        nodes = self._rerank_nodes(
            nodes=nodes,
            query_bundle=query_bundle
        )
        return nodes

    def _rerank_nodes(
        self, 
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        if self.reranker:
            nodes = self.reranker.get_ranker(nodes, query_bundle)
        return nodes