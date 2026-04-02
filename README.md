# Sentiment Trading Platform

Plateforme d'analyse de sentiment financier pour traders.
Deux modules : Twitter scraping + NLP documents → signaux actionnables + backtest.

## Statut des features

| Feature | Statut |
|---|---|
| Structure monorepo + config | ✅ DONE |
| DB SQLite (5 tables + CRUD) | ✅ DONE |
| Shared NLP — VADER | ✅ DONE |
| Shared NLP — FinBERT (singleton) | ✅ DONE |
| Backtest Price Client (yfinance + AV) | ✅ DONE |
| Backtest Engine (3 métriques) | ✅ DONE |
| Twitter — Mock backend | ✅ DONE |
| Twitter — snscrape backend | ✅ DONE |
| Twitter — API backend (tweepy) | ✅ DONE |
| Twitter — Preprocessor | ✅ DONE |
| Twitter — Signal Extractor | ✅ DONE |
| NLP Docs — Parsers (PDF/HTML/TXT) | ✅ DONE |
| NLP Docs — Mock Documents (x6) | ✅ DONE |
| NLP Docs — Chunker | ✅ DONE |
| NLP Docs — NER (spaCy + custom) | ✅ DONE |
| NLP Docs — Pipeline | ✅ DONE |
| NLP Docs — Signal Aggregator | ✅ DONE |
| Dashboard — Tab Twitter Live | ✅ DONE |
| Dashboard — Tab Twitter Backtest | ✅ DONE |
| Dashboard — Tab Document Analysis | ✅ DONE |
| Dashboard — Tab Document Backtest | ✅ DONE |

## Setup

```bash
git clone <repo>
cd sentiment_platform
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env        # remplir les clés si nécessaire
python -c "from shared.db.database import init_db; init_db()"
streamlit run dashboard/app.py
```

## Variables d'environnement (.env)

```env
TWITTER_BACKEND=mock             # mock | snscrape | api
TWITTER_BEARER_TOKEN=            # requis si backend=api
ALPHA_VANTAGE_KEY=               # requis si horizon < 1h en backtest
SENTIMENT_MODEL=vader            # vader | finbert
DB_PATH=shared/db/sentiment.db
```

## Demo rapide

```bash
# Lancer les tests
pytest tests/ -v

# Demo end-to-end en CLI
python -c "
from module1_twitter.twitter.client import TwitterClient
from module1_twitter.nlp.preprocessor import clean_tweet
from module1_twitter.signal.extractor import compute_all_windows
from shared.nlp.vader_sentiment import VaderSentiment
from config import ASSETS

client = TwitterClient('mock')
vader = VaderSentiment()
enriched = []
for asset, cfg in ASSETS.items():
    for t in client.search(cfg['keywords'], limit=50):
        clean = clean_tweet(t['text'])
        r = vader.analyze(clean)
        enriched.append({**t, 'asset_tag': asset, 'sentiment_score': r['score'], 'sentiment_label': r['label']})

signals = compute_all_windows(enriched, 'WTI')
for k, v in signals.items():
    print(f'WTI {k}: {v[\"signal\"]:+.4f} ({v[\"tweet_count\"]} tweets)')
"

# Lancer le dashboard
python -m streamlit run dashboard/app.py
```


