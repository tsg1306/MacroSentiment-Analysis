class SnscrapeBackend:
    """Stub snscrape backend -- not yet implemented."""

    def search(self, keywords, limit=100):
        raise NotImplementedError(
            "snscrape backend not available - use mock or api"
        )

    def get_user_tweets(self, username, limit=10):
        raise NotImplementedError(
            "snscrape backend not available - use mock or api"
        )
