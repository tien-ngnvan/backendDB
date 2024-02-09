"""Base query engine."""

import logging
from abc import abstractmethod
from typing import Any, List, Optional

from src.callbacks.callback_manager import CallbackManager
from src.node.base_node import NodeWithScore
from src.retriever.types import QueryBundle, QueryType

logger = logging.getLogger(__name__)


class BaseQueryEngine:
    """Base query engine."""

    def __init__(
        self, 
        callback_manager: Optional[CallbackManager],
    ) -> None:
        self.callback_manager = callback_manager or CallbackManager([])


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
        raise NotImplementedError(
            "This query engine does not support retrieve, use query directly"
        )

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> Any:
        pass

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> Any:
        pass
