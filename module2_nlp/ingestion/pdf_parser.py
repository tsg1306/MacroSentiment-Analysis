import fitz  # PyMuPDF
from module2_nlp.ingestion.base_parser import BaseParser


class PdfParser(BaseParser):
    def parse(self, source, doc_type="news", published_at=None, **kwargs) -> dict:
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(source)

        pages = [page.get_text() for page in doc]
        text = "\n".join(pages).strip()
        title = doc.metadata.get("title", "") or text[:50].strip()
        doc.close()

        return {
            "title": title,
            "source": source if isinstance(source, str) else "bytes_input",
            "doc_type": doc_type,
            "published_at": published_at,
            "text": text,
        }
