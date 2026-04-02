import random
import uuid
from datetime import datetime, timedelta, timezone
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Authority accounts with realistic follower counts
# ---------------------------------------------------------------------------
AUTHORITY_ACCOUNTS = {
    "realDonaldTrump": 100_000_000,
    "BBCWorld":         20_000_000,
    "Reuters":          15_000_000,
    "zerohedge":         1_200_000,
    "RaoulGMI":          1_000_000,
    "markets":           2_000_000,
    "elerianmohamed":      500_000,
    "elonmusk":         80_000_000,
    "LynAldenContact":     600_000,
    "NourielRoubini":      700_000,
}


def _fake_account():
    """Return (username, followers_count) for a non-authority fake account."""
    return fake.user_name(), random.randint(50, 50_000)


def _rand_ts(base: datetime, spread_hours: float = 48.0) -> datetime:
    """Return a random UTC datetime within spread_hours before base."""
    offset = random.random() * spread_hours * 3600
    return base - timedelta(seconds=offset)


def _make_tweet(author: str, followers: int, text: str, base_ts: datetime) -> dict:
    return {
        "id": uuid.uuid4().hex[:16],
        "created_at": _rand_ts(base_ts),
        "author": author,
        "followers_count": followers,
        "text": text,
        "retweet_count": random.randint(0, 80_000) if followers > 500_000 else random.randint(0, 200),
        "like_count": random.randint(0, 200_000) if followers > 500_000 else random.randint(0, 500),
    }


# ---------------------------------------------------------------------------
# Scenario tweet generators
# ---------------------------------------------------------------------------

_S1_SIGNAL = [
    "BREAKING: Iran and US reach historic ceasefire agreement. Oil markets expected to react sharply. #WTI #crude",
    "Ceasefire between Iran and US could flood the market with Iranian crude oil. Bearish for WTI prices.",
    "Iran ceasefire deal done. Sanctions relief imminent. Expect petroleum supply surge. #oil",
    "Markets digest Iran-US ceasefire. WTI futures dropping on supply expectations.",
    "Iranian oil back on the market after ceasefire. OPEC balance disrupted.",
    "Crude oil prices slide as Iran ceasefire removes geopolitical premium. Pipeline flows resuming.",
    "WTI drops 4% on Iran-US ceasefire news. Refinery margins also under pressure.",
    "This ceasefire changes everything for crude oil supply dynamics. Bearish outlook.",
    "Iran deal is huge for oil. Expect barrel prices to fall significantly in coming weeks.",
    "Oil bears celebrating the Iran ceasefire. WTI headed lower.",
]

_S1_NOISE = [
    "Just had the best coffee this morning, nothing like a fresh brew.",
    "Traffic is terrible today, anyone else stuck on the highway?",
    "New season of my favorite show just dropped, binge watching tonight!",
    "Weather forecast says rain all week, better grab an umbrella.",
    "My cat just knocked over my coffee mug, classic Monday.",
]

_S2_SIGNAL = [
    "OPEC announces surprise production cut of 2M barrels/day. Bullish for crude oil! #WTI #OPEC",
    "OPEC+ agrees to massive output cut. Oil prices set to surge. #crude",
    "Energy markets rally as OPEC slashes production. WTI futures up 5%.",
    "OPEC cut is bigger than expected. Barrel prices jumping. Bullish crude.",
    "Oil supply tightening fast after OPEC decision. Petroleum stocks drawing down.",
    "Refinery margins expanding on OPEC cut. Pipeline operators bullish.",
    "WTI breaking out on OPEC production cut news. Energy stocks following.",
    "OPEC flexing muscles again. Oil bulls in control of the market.",
    "Crude oil surging as OPEC delivers on promised cuts. Sanctions amplifying effect.",
    "Oil traders scrambling after OPEC cut. Barrel premium rising fast.",
]

_S2_NOISE = [
    "Solar energy adoption is accelerating globally, exciting times ahead.",
    "Electric vehicles are getting cheaper every quarter.",
    "Had a great workout at the gym today, feeling energized.",
    "Anyone recommend a good podcast about technology?",
    "Weekend plans: hiking and BBQ with friends.",
]

_S3_SIGNAL = [
    "US inflation comes in way above expectations. CPI at 6.2%. SPX futures tanking. #recession",
    "Inflation surprise! Fed will have to tighten more aggressively. Bearish equities, bullish gold.",
    "S&P 500 drops on hot inflation data. Stocks selling off across the board. #SPX",
    "Gold surges as inflation hedge demand spikes. XAU/USD breaking resistance. #gold #inflation",
    "CPI shocker: inflation still running hot. Recession fears mounting. Sell equities.",
    "Fed rate hike expectations surge after inflation data. SPX bearish outlook.",
    "Gold is the safe haven play here. Inflation eroding purchasing power. #bullion #XAU",
    "Equities crushed by inflation data. S&P heading lower. Risk off mode.",
    "Hot inflation = more Fed tightening = bearish stocks. Gold is the trade.",
    "Inflation hedge demand pushing gold to new highs. Bullion dealers reporting shortages.",
]

_S3_NOISE = [
    "Learning to cook Italian food this weekend, any tips?",
    "Just finished reading an amazing book on behavioral economics.",
    "My garden is finally blooming, spring is here!",
    "Thinking about getting a new laptop, any recommendations?",
    "Movie night tonight, popcorn is ready!",
]

_S4_SIGNAL = [
    "Russia escalates military tensions. Gold spiking as safe haven demand surges. #gold #geopolitics",
    "Geopolitical risk premium rising on Russia news. Gold and bullion demand through the roof.",
    "Gold hits all-time high as Russia tensions intensify. XAU breaking out. #safehaven",
    "Russian conflict fears drive gold demand. Investors fleeing to safety.",
    "Tensions with Russia pushing gold higher. Safe haven flows accelerating.",
    "XAU/USD surging on Russia escalation. War premium being priced in.",
    "Gold traders positioning for extended Russia conflict. Bullion demand soaring.",
    "Russia sanctions talk boosting gold. Sanctions disrupting supply chains.",
    "Geopolitical uncertainty at extreme levels. Gold is the only safe haven left.",
    "Russia crisis deepening. Gold benefits from flight to safety. #XAU",
]

_S4_NOISE = [
    "Just adopted a puppy, life is good!",
    "Starting a new online course on data science.",
    "Beautiful sunset tonight, nature never disappoints.",
    "Trying out a new restaurant downtown this evening.",
    "Finally organized my home office, productivity boost incoming.",
]


def _build_scenario(signal_texts, noise_texts, authority_map, base_ts, n_total=50):
    """Build tweets for one scenario.  ~30 % noise."""
    tweets = []
    n_noise = max(1, int(n_total * 0.30))
    n_signal = n_total - n_noise

    # authority tweets (signal)
    authority_items = list(authority_map.items())
    for i in range(min(len(authority_items), n_signal)):
        author, followers = authority_items[i % len(authority_items)]
        text = signal_texts[i % len(signal_texts)]
        tweets.append(_make_tweet(author, followers, text, base_ts))

    # fill remaining signal with fake accounts
    for i in range(len(authority_items), n_signal):
        user, foll = _fake_account()
        text = signal_texts[i % len(signal_texts)]
        tweets.append(_make_tweet(user, foll, text, base_ts))

    # noise tweets from fake accounts
    for i in range(n_noise):
        user, foll = _fake_account()
        text = noise_texts[i % len(noise_texts)]
        tweets.append(_make_tweet(user, foll, text, base_ts))

    return tweets


class MockBackend:
    """Generates ~200 synthetic tweets across 4 macro scenarios."""

    def __init__(self):
        base_ts = datetime.now(timezone.utc)
        self._tweets = []

        # S1: Iran/US ceasefire -> WTI bearish
        s1_auth = {
            "realDonaldTrump": 100_000_000,
            "BBCWorld": 20_000_000,
            "Reuters": 15_000_000,
        }
        self._tweets.extend(_build_scenario(_S1_SIGNAL, _S1_NOISE, s1_auth, base_ts, n_total=50))

        # S2: OPEC cut -> WTI bullish
        s2_auth = {
            "zerohedge": 1_200_000,
            "RaoulGMI": 1_000_000,
        }
        self._tweets.extend(_build_scenario(_S2_SIGNAL, _S2_NOISE, s2_auth, base_ts, n_total=50))

        # S3: US inflation surprise -> SPX bearish, Gold bullish
        s3_auth = {
            "markets": 2_000_000,
            "elerianmohamed": 500_000,
        }
        self._tweets.extend(_build_scenario(_S3_SIGNAL, _S3_NOISE, s3_auth, base_ts, n_total=50))

        # S4: Russia tensions -> Gold spike
        s4_auth = {
            "elonmusk": 80_000_000,
            "LynAldenContact": 600_000,
            "NourielRoubini": 700_000,
        }
        self._tweets.extend(_build_scenario(_S4_SIGNAL, _S4_NOISE, s4_auth, base_ts, n_total=50))

        random.shuffle(self._tweets)

    def search(self, keywords, limit=100):
        """Filter tweets matching any keyword (case-insensitive), return up to limit."""
        kw_lower = [k.lower() for k in keywords]
        results = [
            t for t in self._tweets
            if any(kw in t["text"].lower() for kw in kw_lower)
        ]
        return results[:limit]

    def get_user_tweets(self, username, limit=10):
        """Filter tweets by author, return up to limit."""
        results = [t for t in self._tweets if t["author"] == username]
        return results[:limit]
