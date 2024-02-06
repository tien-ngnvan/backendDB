import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import fsspec

from src.constants import (
    DOC_STORE_KEY,
)

from src.storage.docstore.simple_store import SimpleDocumentStore
from src.storage.docstore.base import (
    DEFAULT_PERSIST_FNAME as DOCSTORE_FNAME,
    BaseDocumentStore
)
from src.utils.files import concat_dirs
from src.vector_stores.base_vector import VectorStore

DEFAULT_PERSIST_DIR = "./storage"


@dataclass
class StorageContext:
    """Storage context.

    The storage context container is a utility container for storing nodes,
    indices, and vectors. It contains the following:
    - docstore: BaseDocumentStore
    - index_store: BaseIndexStore
    - vector_store: VectorStore

    """

    docstore: BaseDocumentStore
    vector_stores: Dict[str, VectorStore]

    @classmethod
    def from_defaults(
        cls,
        docstore: Optional[BaseDocumentStore] = None,
        persist_dir: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "StorageContext":
        """Create a StorageContext from defaults.

        Args:
            docstore (Optional[BaseDocumentStore]): document store
            index_store (Optional[BaseIndexStore]): index store
            vector_store (Optional[VectorStore]): vector store
            graph_store (Optional[GraphStore]): graph store
            image_store (Optional[VectorStore]): image store

        """
        if persist_dir is None:
            docstore = docstore or SimpleDocumentStore()

        
        else:
            docstore = docstore or SimpleDocumentStore.from_persist_dir(
                persist_dir, fs=fs
            )

        return cls(
            docstore=docstore,
        )

    def persist(
        self,
        persist_dir: Union[str, os.PathLike] = DEFAULT_PERSIST_DIR,
        docstore_fname: str = DOCSTORE_FNAME,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the storage context.

        Args:
            persist_dir (str): directory to persist the storage context
        """
        if fs is not None:
            persist_dir = str(persist_dir)  # NOTE: doesn't support Windows here
            docstore_path = concat_dirs(persist_dir, docstore_fname)
        else:
            persist_dir = Path(persist_dir)
            docstore_path = str(persist_dir / docstore_fname)

        self.docstore.persist(persist_path=docstore_path, fs=fs)
    

    def to_dict(self) -> dict:
        all_simple = (
            isinstance(self.docstore, SimpleDocumentStore)
        )
        if not all_simple:
            raise ValueError(
                "to_dict only available when using simple doc/index/vector stores"
            )

        assert isinstance(self.docstore, SimpleDocumentStore)

        return {
            DOC_STORE_KEY: self.docstore.to_dict(),
        }

    @classmethod
    def from_dict(cls, save_dict: dict) -> "StorageContext":
        """Create a StorageContext from dict."""
        docstore = SimpleDocumentStore.from_dict(save_dict[DOC_STORE_KEY])

        return cls(
            docstore=docstore,
        )