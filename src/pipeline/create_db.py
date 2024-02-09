import logging 
from typing import Optional, List
import time

from src.config.schema import (
    MilvusConfig,
    MilvusArguments,
    IndexRetrieveParams
)

from src.core.storage_context import StorageContext
from src.core.service_context import ServiceContext
from src.node_parser.vi_normalizer import ViNormalizer
from src.node_parser.base import NodeParser
from src.vector_stores.milvus import MilvusVectorStore
from src.engine.data_ingestion import DatabaseEngine
from src.callbacks.callback_manager import CallbackManager

logger = logging.getLogger(__name__)

class InitializeDatabase:
    def __init__(self,
        milvus_config: MilvusConfig,
        milvus_params: MilvusArguments,
        index_params: IndexRetrieveParams,
        service_context: Optional[ServiceContext],
        callback_manager: CallbackManager,
    ) -> None:
        self.milvus_config = milvus_config
        self.milvus_params = milvus_params
        self.index_params = index_params
        self.service_context = service_context      

        self.parser: NodeParser = None
        self.normalizer: ViNormalizer = None

        self.callback_manager = callback_manager

    def main(self, documents):

        """Wrap time"""
        start_build_collection_index = int(round(time.time() * 1000))

        #TODO: build milvus vector from documents
        milvus_vector_store = MilvusVectorStore(self.milvus_config, self.milvus_params)

        # construct index and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=milvus_vector_store
        )

        docs = [self.normalizer.normalize(doc) for doc in documents]

        nodes = self.parser.get_nodes_from_documents(
            documents=docs,
            show_progress=True,
        )

        # TODO:
        # (1) Add build index from nodes to get embeddings
        data_ingestor = DatabaseEngine(
            insert_batch_size=self.index_params.insert_batch_size,
            callback_manager=self.callback_manager,
            splitter=self.normalizer,
            service_context=self.service_context,
            storage_context=storage_context,
        )

        data_ingestor.run_engine(nodes=nodes, show_progress=True)

        storage_context.persist()
        end_build_collection_index = int(round(time.time() * 1000))
        print(f"Time for build collection and index: {end_build_collection_index - start_build_collection_index} ms")

        return "Add success!!"

    