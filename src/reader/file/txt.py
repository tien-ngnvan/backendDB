from rag.reader.base_reader import BaseReader
from rag.node.base_node import Document


class TxtReader(BaseReader):
    """ Txt parser """

    def load_data(self, file, extra_info=None):
        # load_data returns a list of Document objects
        with open(file, "r") as f:
            text = f.read()
        return [Document(text=text, extra_info=extra_info or {})]