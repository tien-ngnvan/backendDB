from omegaconf import DictConfig

from src.constants import *
from src.utils.files import (
    create_directories
)
from .schema import (
    OtherConfig,
    MilvusConfig,
    CassandraConfig,
    SentenceSplitterConfig,
    RecursiveSplitterConfig,
    SbertConfig,
    CrossEmbeddingConfig,
    AsymRerankConfig,
    CohereRerankConfig
)

class ConfigurationManager: 
    def __init__(
        self, 
        config: DictConfig, 
    ) -> None:
        self.config = config
        self.other_config = self.get_orther_config()
        
        create_directories([self.other_config.artifacts_root])
    
    def get_milvus_config(self) -> MilvusConfig: 
        """create instace for vector store config"""
        milvus_config = MilvusConfig(
            vectorstore_name= self.config.vector_store.vectorstore_name,
            embedding_dim=self.config.vector_store.embedding_dim,
            host= self.config.vector_store.host,
            port= self.config.vector_store.port,
            address= self.config.vector_store.address,
            uri= self.config.vector_store.uri,
            user= self.config.vector_store.user,
            consistency_level=self.config.vector_store.consistency_level,
            primary_field=self.config.vector_store.primary_field,
            text_field=self.config.vector_store.text_field,
            embedding_field=self.config.vector_store.embedding_field,
            collection_name=self.config.vector_store.collection_name,
            index_params=self.config.vector_store.index_params,
            search_params=self.config.vector_store.search_params,
            overwrite=self.config.vector_store.overwrite,
        )
        return milvus_config
    
    def get_cassandra_config(self) -> CassandraConfig: 
        """create instace for vector store config"""
        cassandra_config = CassandraConfig(
            vectorstore_name= self.config.vector_store.vectorstore_name,
            embedding_dim= self.config.vector_store.embedding_dim,
            table= self.config.vector_store.table,
            session= self.config.vector_store.session,
            keyspace= self.config.vector_store.keyspace,
            ttl_seconds= self.config.vector_store.ttl_seconds,
        )
        return cassandra_config
    
    
    def get_sentence_splitter_config(self) -> SentenceSplitterConfig: 
        """create instace for splitter config"""
        sentence_splitter_config = SentenceSplitterConfig(
            splitter_mode= self.config.splitter.splitter_mode,
            model_name_tokenizer= self.config.splitter.model_name_tokenizer,
            separator= self.config.splitter.separator,
            chunk_size= self.config.splitter.chunk_size,
            chunk_overlap= self.config.splitter.chunk_overlap,
            paragraph_separator= self.config.splitter.paragraph_separator,
            secondary_chunking_regex= self.config.splitter.secondary_chunking_regex,
        )
        return sentence_splitter_config

    def get_recursive_splitter_config(self) -> RecursiveSplitterConfig: 
        """create instace for splitter config"""
        recursive_splitter_config = RecursiveSplitterConfig(
            splitter_mode= self.config.splitter.splitter_mode,
            model_name_tokenizer= self.config.splitter.model_name_tokenizer,
            separator= self.config.splitter.separator,
            chunk_size= self.config.splitter.chunk_size,
            chunk_overlap= self.config.splitter.chunk_overlap,
            backup_separators= self.config.splitter.backup_separators,
        )
        return recursive_splitter_config
     
    def get_cross_embed_config(self) -> CrossEmbeddingConfig:
        """create instace for cross embedding config"""
        embed_config = CrossEmbeddingConfig(
            qry_model_name= self.config.embedding.qry_model_name,
            psg_model_name= self.config.embedding.psg_model_name,
            token= self.config.embedding.token,
            proxies= self.config.embedding.proxies,
            methods= self.config.embedding.methods,
            device= self.config.embedding.device,
            embedding_batch_size= self.config.embedding.embedding_batch_size,
            pooling= self.config.embedding.pooling,
            max_length= self.config.embedding.max_length,
            normalize= self.config.embedding.normalize,
        )
        return embed_config

    def get_sbert_embed_config(self) -> SbertConfig:
        """create instace for sentence embedding config"""
        embed_config = SbertConfig(
            model_name= self.config.embedding.model_name, 
            tokenizer_name= self.config.embedding.tokenizer_name,
            pooling= self.config.embedding.pooling,
            max_length= self.config.embedding.max_length,
            normalize= self.config.embedding.normalize,
            embedding_batch_size= self.config.embedding.embedding_batch_size,
            cache_folder= self.config.embedding.cache_folder,
            trust_remote_code= self.config.embedding.trust_remote_code,
        )
        return embed_config
    
    def get_asym_rerank_config(self) -> AsymRerankConfig:
        asym_rerank_config = AsymRerankConfig(
            model_name_or_path=self.config.model_name_or_path,
            token=self.config.token,
            device=self.config.device,
            methods=self.config.methods,
            proxies=self.config.proxies,
        )
        return asym_rerank_config
    
    def get_cohere_config(self) -> CohereRerankConfig:
        cohere_config = CohereRerankConfig(
            top_n=self.config.top_n,
            model=self.config.model,
            api_key=self.config.api_key,
        )
        return cohere_config

    
    def get_orther_config(self) -> OtherConfig:
        """create instace for other config"""
        other_config = OtherConfig(
            artifacts_root= self.config.artifacts_root,
            file_data_path= self.config.file_data_path,
            language= self.config.language,
            use_async= self.config.use_async,
            show_progress= self.config.show_progress,
            vector_store_query_mode= self.config.vector_store_query_mode,
            store_nodes_override= self.config.store_nodes_override,
            insert_batch_size= self.config.insert_batch_size,
            similarity_top_k= self.config.similarity_top_k,
            use_rerank= self.config.use_rerank,
        )
        return other_config