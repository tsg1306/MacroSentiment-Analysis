import os
import sys
import hashlib
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data_tweet', 'financial_juice_tweets.csv'
)

_DATE_FMT = "%a %b %d %H:%M:%S +0000 %Y"


def _parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s.strip(), _DATE_FMT)
    except Exception:
        return datetime.utcnow()


def _make_id(row) -> str:
    raw = f"{row['date']}{row['content']}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


class CsvBackend:
    def __init__(self):
        path = os.path.abspath(CSV_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tweet CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        self._tweets = []
        for _, row in df.iterrows():
            text = str(row.get('content', '')).strip()
            if not text or text.startswith('http'):
                continue
            self._tweets.append({
                "id":              _make_id(row),
                "created_at":      _parse_date(str(row.get('date', ''))),
                "author":          str(row.get('author_name', 'FinancialJuice')),
                "followers_count": 0,
                "text":            text,
                "retweet_count":   0,
                "like_count":      0,
            })

    def search(self, keywords: list, limit: int = 100) -> list:
        if not keywords:
            return self._tweets[:limit]
        kw_lower = [k.lower() for k in keywords]
        results = [
            t for t in self._tweets
            if any(kw in t["text"].lower() for kw in kw_lower)
        ]
        return results[:limit]

    def get_user_tweets(self, username: str, limit: int = 10) -> list:
        results = [t for t in self._tweets if t["author"].lower() == username.lower()]
        return results[:limit]

    def get_all(self) -> list:
        """Returns all tweets — used by dashboard for full pipeline."""
        return self._tweets
