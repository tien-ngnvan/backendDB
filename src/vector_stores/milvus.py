"""Milvus vector store index.

An index that is built within Milvus.

"""
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from src.node.base_node import BaseNode, TextNode
from src.vector_stores import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from src.config.schema import MilvusArguments, MilvusConfig
from .utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

if TYPE_CHECKING:
    from pymilvus import MilvusClient

logger = logging.getLogger(__name__)

MILVUS_ID_FIELD = "id"


def _to_milvus_filter(standard_filters: MetadataFilters) -> List[str]:
    """Translate standard metadata filters to Milvus specific spec."""
    filters = []
    for filter in standard_filters.legacy_filters():
        if isinstance(filter.value, str):
            filters.append(str(filter.key) + " == " + '"' + str(filter.value) + '"')
        else:
            filters.append(str(filter.key) + " == " + str(filter.value))
    return filters


class MilvusVectorStore(VectorStore):
    """The Milvus Vector Store.

    In this vector store we store the text, its embedding and
    a its metadata in a Milvus collection. This implementation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection doesn't
    exist or if `overwrite` is set to True.

    Args:
        uri (str, optional): The URI to connect to, comes in the form of
            "http://address:port".
        token (str, optional): The token for log in. Empty if not using rbac, if
            using rbac it will most likely be "username:password".
        collection_name (str, optional): The name of the collection where data will be
            stored. Defaults to "llamalection".
        dim (int, optional): The dimension of the embedding vectors for the collection.
            Required if creating a new collection.
        embedding_field (str, optional): The name of the embedding field for the
            collection, defaults to DEFAULT_EMBEDDING_KEY.
        doc_id_field (str, optional): The name of the doc_id field for the collection,
            defaults to DEFAULT_DOC_ID_KEY.
        similarity_metric (str, optional): The similarity metric to use,
            currently supports IP and L2.
        consistency_level (str, optional): Which consistency level to use for a newly
            created collection. Defaults to "Session".
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.
        text_key (str, optional): What key text is stored in in the passed collection.
            Used when bringing your own collection. Defaults to None.
        index_config (dict, optional): The configuration used for building the
            Milvus index. Defaults to None.
        search_config (dict, optional): The configuration used for searching
            the Milvus index. Note that this must be compatible with the index
            type specified by `index_config`. Defaults to None.

    Raises:
        ImportError: Unable to import `pymilvus`.
        MilvusException: Error communicating with Milvus, more can be found in logging
            under Debug.

    Returns:
        MilvusVectorstore: Vectorstore that supports add, delete, and query.
    """

    stores_text: bool = True
    stores_node: bool = True

    def __init__(
        self,
        config: MilvusConfig,
        params: MilvusArguments,
    ) -> None:
        """Init params."""
        import_err_msg = (
            "`pymilvus` package not found, please run `pip install pymilvus`"
        )
        try:
            import pymilvus  # noqa
        except ImportError:
            raise ImportError(import_err_msg)

        from pymilvus import Collection

        self.config = config
        self.params = params

        self.embedding_field = config.embedding_field
        self.doc_id_field = config.primary_field
        self.text_field = config.text_field
        self.consistency_level = config.consistency_level
        self.dim = params.embedding_dim
        self.collection_name = params.collection_name
        self.overwrite = params.overwrite
        

        # Select the similarity metric
        self.similarity_metric = self.params.search_params.get("metric_type", None)

        # Connect to Milvus instance
        self.milvusclient = self.connect_client()

        # Delete previous collection if overwriting
        if self.overwrite and self.collection_name in self.milvusclient.list_collections():
            self.milvusclient.drop_collection(self.collection_name)

        # Create the collection if it does not exist
        if self.collection_name not in self.milvusclient.list_collections():
            if self.dim is None:
                raise ValueError("Dim argument required for collection creation.")
            self.milvusclient.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                primary_field_name=MILVUS_ID_FIELD,
                vector_field_name=self.embedding_field,
                id_type="string",
                metric_type=self.similarity_metric,
                max_length=65_535,
            )
        
        self.collection = Collection(
            self.collection_name, using=self.milvusclient._using
        )
        self._create_index_if_required(force=True)
        logger.debug(f"Successfully created a new collection: {self.collection_name}")

    def connect_client(self) -> "MilvusClient":
        from pymilvus import MilvusClient
        # Order of use is host/port, uri, address
        if self.config.uri is not None:
            return MilvusClient(self.config.uri)
        elif self.config.host is not None and self.config.port is not None:
            uri = f"http://{self.config.host}:{self.config.port}"
            return MilvusClient(uri=uri)
        raise Exception("Cannot connect milvus database")


    @property
    def client(self) -> Any:
        """Get client."""
        return self.milvusclient

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add the embeddings and their nodes into Milvus.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.
        """
        insert_list = []
        insert_ids = []

        # Process that data we are going to insert
        for node in nodes:
            entry = node_to_metadata_dict(node)
            entry[MILVUS_ID_FIELD] = node.node_id
            entry[self.embedding_field] = node.embedding

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        # Insert the data into milvus
        self.collection.insert(insert_list)
        self.collection.flush()
        self._create_index_if_required()
        logger.debug(
            f"Successfully inserted embeddings into: {self.collection_name} "
            f"Num Inserted: {len(insert_list)}"
        )
        return insert_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        Raises:
            MilvusException: Failed to delete the doc.
        """
        # Adds ability for multiple doc delete in future.
        doc_ids: List[str]
        if isinstance(ref_doc_id, list):
            doc_ids = ref_doc_id  # type: ignore
        else:
            doc_ids = [ref_doc_id]

        # Begin by querying for the primary keys to delete
        doc_ids = ['"' + entry + '"' for entry in doc_ids]
        entries = self.milvusclient.query(
            collection_name=self.collection_name,
            filter=f"{self.doc_id_field} in [{','.join(doc_ids)}]",
        )
        ids = [entry["id"] for entry in entries]
        self.milvusclient.delete(collection_name=self.collection_name, pks=ids)
        logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        expr = []
        output_fields = ["*"]

        # Parse the filter
        if query.filters is not None:
            expr.extend(_to_milvus_filter(query.filters))

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

        # Limit output fields
        if query.output_fields is not None:
            output_fields = query.output_fields

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = " and ".join(expr)

        # Perform the search
        res = self.milvusclient.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=self.params.search_params,
        )

        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )

        nodes = []
        similarities = []
        ids = []

        # Parse the results
        for hit in res[0]:
            if not self.text_key:
                node = metadata_dict_to_node(
                    {
                        "_node_content": hit["entity"].get("_node_content", None),
                        "_node_type": hit["entity"].get("_node_type", None),
                    }
                )
            else:
                try:
                    text = hit["entity"].get(self.text_key)
                except Exception:
                    raise ValueError(
                        "The passed in text_key value does not exist "
                        "in the retrieved entity."
                    )
                node = TextNode(
                    text=text,
                )
            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(hit["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _create_index_if_required(self, force: bool = False) -> None:
        """
        This method is introduced to allow the index to be created
        both in the constructor and in the `add` method. 
        (1) The `force` flag is provided to ensure that the index is 
        created in the constructor evenif self.overwrite is false. 
        (2) In the `add` method, the index is re-created only 
        if self.overwrite is true.
        """

        if (self.collection.has_index() and self.overwrite) or force:
            # release out of memory first
            self.collection.release()
            # drop index of that field before build new overwrite index
            self.collection.drop_index()
            # set params
            index_params: Dict[str, Union[str, Dict[str, Any]]] = self.params.index_params
            self.collection.create_index(
                self.embedding_field, index_params=index_params
            )
            print("Create index succesfully")
            self.collection.load()