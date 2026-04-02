from datetime import datetime
from bs4 import BeautifulSoup
from module2_nlp.ingestion.base_parser import BaseParser


class HtmlParser(BaseParser):
    def parse(self, source, doc_type="news", **kwargs) -> dict:
        soup = BeautifulSoup(source, "html.parser")

        # Remove non-content tags
        for tag in soup.find_all(["nav", "footer", "script", "style"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" ", strip=True)

        # Extract published_at from meta tags
        published_at = None
        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta and meta.get("content"):
            try:
                published_at = datetime.fromisoformat(meta["content"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return {
            "title": title or text[:50].strip(),
            "source": "html_input",
            "doc_type": doc_type,
            "published_at": published_at,
            "text": text,
        }
