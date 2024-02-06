from typing import Any

class ViNormalizer:
    """Abstract Text Normalize protocol."""

    language: str

    def normalize(
        self,
        text: str,
        **add_kwargs: Any,
    ) -> str:
        """Add nodes with embedding to vector store."""
        from underthesea import text_normalize
        return text_normalize(text)

    async def async_normalize(
        self,
        text: str,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronously add nodes with embedding to vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call add synchronously.
        """
        return self.normalize(text)