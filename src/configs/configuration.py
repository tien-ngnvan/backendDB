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
            vectorstore_name= self.config.vectorstore_name,
            embedding_dim=self.config.embedding_dim,
            host= self.config.host,
            port= self.config.port,
            address= self.config.address,
            uri= self.config.uri,
            user= self.config.user,
            consistency_level=self.config.consistency_level,
            primary_field=self.config.primary_field,
            text_field=self.config.text_field,
            embedding_field=self.config.embedding_field,
            collection_name=self.config.collection_name,
            index_params=self.config.index_params,
            search_params=self.config.search_params,
            overwrite=self.config.overwrite,
        )
        return milvus_config
    
    def get_postgres_config(self) -> CassandraConfig: 
        """create instace for vector store config"""
        cassandra_config = CassandraConfig(
            vectorstore_name= self.config.vectorstore_name,
            embedding_dim= self.config.embedding_dim,
            table= self.config.table,
            session= self.config.session,
            keyspace= self.config.keyspace,
            ttl_seconds= self.config.ttl_seconds,
        )
        return cassandra_config
    
    
    def get_sentence_splitter_config(self) -> SentenceSplitterConfig: 
        """create instace for splitter config"""
        sentence_splitter_config = SentenceSplitterConfig(
            splitter_mode= self.config.splitter_mode,
            model_name_tokenizer= self.config.model_name_tokenizer,
            separator= self.config.separator,
            chunk_size= self.config.chunk_size,
            chunk_overlap= self.config.chunk_overlap,
            paragraph_separator= self.config.paragraph_separator,
            secondary_chunking_regex= self.config.secondary_chunking_regex,
        )
        return sentence_splitter_config

    def get_recursive_splitter_config(self) -> RecursiveSplitterConfig: 
        """create instace for splitter config"""
        recursive_splitter_config = RecursiveSplitterConfig(
            splitter_mode= self.config.splitter_mode,
            model_name_tokenizer= self.config.model_name_tokenizer,
            separator= self.config.separator,
            chunk_size= self.config.chunk_size,
            chunk_overlap= self.config.chunk_overlap,
            backup_separators= self.config.backup_separators,
        )
        return recursive_splitter_config
     
    def get_cross_embed_config(self) -> CrossEmbeddingConfig:
        """create instace for cross embedding config"""
        embed_config = CrossEmbeddingConfig(
            qry_model_name= self.config.qry_model_name,
            psg_model_name= self.config.psg_model_name,
            token= self.config.token,
            proxies= self.config.proxies,
            methods= self.config.methods,
            device= self.config.device,
            embedding_batch_size= self.config.embedding_batch_size,
            pooling= self.config.pooling,
            max_length= self.config.max_length,
            normalize= self.config.normalize,
        )
        return embed_config

    def get_index_retriever_params(self) -> SbertConfig:
        """create instace for sentence embedding config"""
        embed_config = SbertConfig(
            model_name= self.config.model_name, 
            tokenizer_name= self.config.tokenizer_name,
            pooling= self.config.pooling,
            max_length= self.config.max_length,
            normalize= self.config.normalize,
            embedding_batch_size= self.config.embedding_batch_size,
            cache_folder= self.config.cache_folder,
            trust_remote_code= self.config.trust_remote_code,
        )
        return embed_config
    
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
        )
        return other_config