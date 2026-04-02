from module2_nlp.ingestion.base_parser import BaseParser


class TxtParser(BaseParser):
    def parse(self, source, title=None, doc_type="news", published_at=None, **kwargs) -> dict:
        text = source if isinstance(source, str) else source.decode("utf-8")
        return {
            "title": title or text[:50].strip(),
            "source": "text_input",
            "doc_type": doc_type,
            "published_at": published_at,
            "text": text.strip(),
        }
