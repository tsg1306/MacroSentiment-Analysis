import os
from dotenv import load_dotenv
load_dotenv()

# ── Twitter ───────────────────────────────────────────────────────────────────
TWITTER_BACKEND = os.getenv("TWITTER_BACKEND", "mock")   # "mock" | "snscrape" | "api"
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

HIGH_AUTHORITY_ACCOUNTS = [
    "realDonaldTrump", "elonmusk", "zerohedge", "RaoulGMI",
    "LynAldenContact", "elerianmohamed", "NourielRoubini", "markets"
]

# ── Assets (Module 1) ─────────────────────────────────────────────────────────
ASSETS = {
    "WTI":   {"keywords": ["oil","crude","WTI","petroleum","OPEC","barrel",
                            "ceasefire","Iran","sanctions","pipeline","refinery"],
              "yf_ticker": "CL=F",   "av_symbol": "USOIL"},
    "BRENT": {"keywords": ["Brent","North Sea","oil","crude"],
              "yf_ticker": "BZ=F",   "av_symbol": "UKOIL"},
    "GOLD":  {"keywords": ["gold","XAU","bullion","safe haven","inflation hedge"],
              "yf_ticker": "GC=F",   "av_symbol": "XAUUSD"},
    "SPX":   {"keywords": ["S&P","SPX","equities","stocks","recession","bull market"],
              "yf_ticker": "^GSPC",  "av_symbol": "SPX"},
}

# ── Entity → Ticker (Module 2 backtest) ───────────────────────────────────────
ENTITY_TICKERS = {
    "oil":     {"yf": "CL=F",     "av": "USOIL"},
    "crude":   {"yf": "CL=F",     "av": "USOIL"},
    "gold":    {"yf": "GC=F",     "av": "XAUUSD"},
    "SPX":     {"yf": "^GSPC",    "av": "SPX"},
    "NASDAQ":  {"yf": "^IXIC",    "av": "NDX"},
    "EUR/USD": {"yf": "EURUSD=X", "av": "EURUSD"},
}

# ── NLP ───────────────────────────────────────────────────────────────────────
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "vader")   # "vader" | "finbert"

TRACKED_ENTITIES = {
    "commodities": ["oil","crude","WTI","Brent","gold","XAU",
                    "natural gas","copper","wheat","silver"],
    "macro":       ["inflation","recession","interest rates","Fed","ECB",
                    "GDP","unemployment","CPI","yield curve"],
    "equities":    ["S&P","SPX","NASDAQ","equities","stocks","earnings"],
    "geopolitics": ["China","Russia","Iran","OPEC","sanctions",
                    "tariffs","war","ceasefire","conflict"],
}

# ── Module 2 chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE = 200             # mots par chunk
CHUNK_OVERLAP = 50           # overlap entre chunks
ENTITY_CONTEXT_WINDOW = 100  # tokens contexte autour d'une entité

# ── Signal ────────────────────────────────────────────────────────────────────
SIGNAL_WINDOWS = [60, 240, 1440]   # fenêtres temps réel module 1 (minutes : 1h, 4h, 24h)
ALERT_THRESHOLD = 0.4

# ── Backtest ──────────────────────────────────────────────────────────────────
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
INTRADAY_THRESHOLD_HOURS = 1.0   # < ce seuil → Alpha Vantage, sinon yfinance

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "shared/db/sentiment.db")
