from typing import Any

class ViNormalizer:
    """Abstract Text Normalize protocol."""

    language: str

    def __init__(self) -> None:
        pass

    def normalize(
        self,
        text: str,
        **add_kwargs: Any,
    ) -> str:
        """Add nodes with embedding to vector store."""
        from underthesea import text_normalize
        #TODO:
        """
        (1) copy cái emotion -> gán qua đây
        (2) features:
            - remove icon, (có sẵn)
            - remove bullet (research)
            - html tags:  </p> (research)
            - remove multiple space: strip() (có sẵn)
        (3) specify type:
            - chỉnh lại bên en
            - format bên vi: chỉnh init, chỉnh type input output của def
        """
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