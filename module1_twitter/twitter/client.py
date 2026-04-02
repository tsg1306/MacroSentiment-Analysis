import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TwitterClient:
    def __init__(self, backend="mock"):
        if backend == "mock":
            from module1_twitter.twitter.mock_backend import MockBackend
            self.backend = MockBackend()
        elif backend == "snscrape":
            from module1_twitter.twitter.snscrape_backend import SnscrapeBackend
            self.backend = SnscrapeBackend()
        elif backend == "api":
            from module1_twitter.twitter.api_backend import ApiBackend
            self.backend = ApiBackend()
        elif backend == "csv":
            from module1_twitter.twitter.csv_backend import CsvBackend
            self.backend = CsvBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def search(self, keywords, limit=100):
        return self.backend.search(keywords, limit)

    def get_user_tweets(self, username, limit=10):
        return self.backend.get_user_tweets(username, limit)

    def get_all(self) -> list:
        """Returns all tweets — only available for csv backend."""
        if hasattr(self.backend, 'get_all'):
            return self.backend.get_all()
        return []
