import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import TWITTER_BEARER_TOKEN


class ApiBackend:
    """Twitter API v2 backend using tweepy.

    The __init__ validates that TWITTER_BEARER_TOKEN is set.
    Actual API calls are stubbed with NotImplementedError for now.
    """

    def __init__(self):
        if not TWITTER_BEARER_TOKEN:
            raise ValueError(
                "TWITTER_BEARER_TOKEN is empty. "
                "Set it in your .env file to use the API backend."
            )
        # Will initialise tweepy.Client here once token is available
        self._token = TWITTER_BEARER_TOKEN

    def search(self, keywords, limit=100):
        raise NotImplementedError(
            "Twitter API search not yet implemented. "
            "Set TWITTER_BEARER_TOKEN and install tweepy."
        )

    def get_user_tweets(self, username, limit=10):
        raise NotImplementedError(
            "Twitter API get_user_tweets not yet implemented. "
            "Set TWITTER_BEARER_TOKEN and install tweepy."
        )
