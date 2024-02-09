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
from src.engine.retrieve_engine import RetriverEngine
from src.vector_stores.milvus import MilvusVectorStore
from llama_index.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)

class RetrieverPipeline:
    def __init__(self,
        milvus_config: MilvusConfig,
        milvus_params: MilvusArguments,
        index_params: IndexRetrieveParams,
        service_context: Optional[ServiceContext],
        node_postprocessors: Optional[List[BaseNodePostprocessor]],
    ) -> None:
        self.milvus_config = milvus_config
        self.milvus_params = milvus_params
        self.index_params = index_params
        self.service_context = service_context
        self.node_postprocessors = node_postprocessors                    

    def main(self, query, documents):

        """Wrap time"""
        start_build_collection_index = int(round(time.time() * 1000))

        #TODO: build milvus vector from documents 
        milvus_vector_store = MilvusVectorStore(self.milvus_config, self.milvus_params)
        # construct index and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=milvus_vector_store
        )
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            service_context=self.service_context,
            store_nodes_override=self.index_params.store_nodes_override,
            insert_batch_size=self.index_params.insert_batch_size,
            use_async=self.index_params.use_async,
            show_progress=self.index_params.show_progress,
        )
        
        end_build_collection_index = int(round(time.time() * 1000))
        print(f"Time for build collection and index: {end_build_collection_index - start_build_collection_index} ms")

        """Wrap for loading retriever and search"""
        start_build_retrieve_search = int(round(time.time() * 1000))
        #TODO: build retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.index_params.similarity_top_k,
            sparse_top_k=self.index_params.sparse_top_k,
            alpha=self.index_params.alpha,
            list_query_mode=self.index_params.list_query_mode,
            keyword_table_mode=self.index_params.keyword_table_mode,
            max_keywords_per_chunk=self.index_params.keyword_table_mode,
            max_keywords_per_query=self.index_params.max_keywords_per_query,
            num_chunks_per_query=self.index_params.num_chunks_per_query,
            vector_store_query_mode=self.index_params.vector_store_query_mode,
        )


        #TODO: build retriever
        retriever = None

        #TODO: rerank
        rerank = None

        #TODO: query
        engine = RetriverEngine(
            retriever=retriever,
            reranker=rerank
        )
        

        #TODO: query
        docs = engine.search(query)

        end_build_retrieve_search = int(round(time.time() * 1000))
        print(f"Time for build retriever and search: {end_build_retrieve_search - start_build_retrieve_search} ms")

        return docs

    def apply_rerank():
        pass