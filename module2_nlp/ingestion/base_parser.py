from abc import ABC, abstractmethod


class BaseParser(ABC):
    @abstractmethod
    def parse(self, source, **kwargs) -> dict:
        """
        Returns:
            {"title": str, "source": str, "doc_type": str,
             "published_at": datetime|None, "text": str}
        """
        pass
