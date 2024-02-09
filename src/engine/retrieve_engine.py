

"""Base query engine."""

import logging
from typing import Any, List, Optional

from src.callbacks.callback_manager import CallbackManager
from src.node.base_node import NodeWithScore
from src.retriever.types import QueryBundle, QueryType
from src.retriever.base_retriver import BaseRetriever

from .base import BaseQueryEngine

logger = logging.getLogger(__name__)

class RetriverEngine(BaseQueryEngine):
    """Base query engine."""

    def __init__(
        self, 
        callback_manager: Optional[CallbackManager],
        retriever: BaseRetriever,
        reranker: Optional[Any] = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        super().__init__(
            callback_manager=callback_manager
        )

    def query(self, str_or_query_bundle: QueryType) -> Any:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return self._query(str_or_query_bundle)

    async def aquery(self, str_or_query_bundle: QueryType) -> Any:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return await self._aquery(str_or_query_bundle)

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self.retriever.retrieve(query_bundle)
        nodes = self._rerank_nodes(
            nodes=nodes,
            query_bundle=query_bundle
        )
        return nodes

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self.retriever.aretrieve(query_bundle)
        nodes = self._rerank_nodes(
            nodes=nodes,
            query_bundle=query_bundle
        )
        return nodes

    def _query(self, query_bundle: QueryBundle) -> Any:
        """Answer a query."""
        return "Not supported"

    async def _aquery(self, query_bundle: QueryBundle) -> Any:
        """Answer a query."""
        return "Not supported"

    def _rerank_nodes(
        self, 
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        if self.reranker:
            nodes = self.reranker.rerank(nodes, query_bundle)
        return nodes