# Macro Intelligence Dashboard

Plateforme d'analyse de sentiment macro pour traders. Combine deux sources de donnees reelles :
- **607 tweets FinancialJuice** (Mar 28-31 2026) — sentiment, topic modeling, consensus/divergence
- **14 PDFs macro** (Goldman Sachs, BofA, Macquarie, Natixis, SEB...) — NER, sentiment par entite, recherche semantique

Dashboard Streamlit unifie avec 4 onglets : Macro Digest, Tweet Intelligence, Corpus Analysis, Backtest.

## Statut des features

| Feature | Statut |
|---|---|
| Structure monorepo + config | ✅ DONE |
| DB SQLite (6 tables + CRUD) | ✅ DONE |
| Shared NLP — VADER + FinBERT (singleton) | ✅ DONE |
| Backtest Engine (yfinance + Alpha Vantage) | ✅ DONE |
| Twitter — CSV backend (FinancialJuice) | ✅ DONE |
| Corpus — ChromaDB store (semantic search) | ✅ DONE |
| Corpus — Ingest PDFs (ChromaDB + SQLite) | ✅ DONE |
| Analysis — BERTopic topic modeling (KMeans) | ✅ DONE |
| Analysis — Consensus/Divergence detection | ✅ DONE |
| Analysis — Cross-source alignment | ✅ DONE |
| Dashboard — Tab Macro Digest | ✅ DONE |
| Dashboard — Tab Tweet Intelligence | ✅ DONE |
| Dashboard — Tab Corpus Analysis | ✅ DONE |
| Dashboard — Tab Backtest | ✅ DONE |
| v2 — Document classification (7 themes TF-IDF) | ✅ DONE |
| v2 — Stance detection (institutional/investor/research) | ✅ DONE |
| v2 — Trade extraction (explicit/implicit) | ✅ DONE |
| v2 — Extractive summarizer (10 bullets, priority) | ✅ DONE |
| v2 — Dashboard rewrite (trades, summaries, filters) | ✅ DONE |

## Quick Start

```bash
# 1. Clone et install
git clone https://github.com/tsg1306/MacroSentiment-Analysis.git
cd sentiment_platform
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configuration
cp .env.example .env
# Editer .env si besoin (Alpha Vantage key pour backtest <1h)

# 3. Init DB + ingest corpus
python -c "from shared.db.database import init_db; init_db()"
python scripts/ingest_corpus.py
# -> Parse 14 PDFs, ~630 chunks dans ChromaDB, ~2786 entites dans SQLite
# -> Classifie chaque doc : domain, stance, trade signals, 10-bullet summary
# -> Duree : 2-5 min selon hardware (NER spaCy + embeddings)

# 4. Lancer le dashboard
streamlit run dashboard/app.py
```

## Dashboard — 4 onglets

### Tab 1 — Macro Digest
Vue synthetique en 60 secondes :
- **5 KPIs** : tweets analyses, docs ingeres, topics detectes, consensus, divergences
- **Heatmap entites x sources** : sentiment par entite (Oil, Gold, Fed, ECB...) croise avec les 14 docs + tweets
- **Digest narratif** : consensus views (top 3), divergences (top 3), signaux faibles
- **Explicit/Implicit Trades** : extraction automatique des recommandations trade depuis les 14 docs
- **Cross-source alignment** : score d'alignement tweets vs corpus par theme (0-1)

### Tab 2 — Tweet Intelligence
Exploration interactive des 607 tweets :
- **Filtres** : topic BERTopic, sentiment, type (consensus/divergence/signal faible)
- **Timeline** : evolution sentiment par tranches de 4h avec pics d'intensite
- **Feed pagine** : 50 tweets/page avec badges et scores
- **Topic Map** : visualisation 2D BERTopic + keywords par topic

### Tab 3 — Corpus Analysis
Analyse des 14 documents macro :
- **Heatmap docs x entites** : score sentiment par document et entite trackee, badges trade [X]/[I]
- **Document summaries** : 10-bullet extractive summary par document, triees par type (explicit > implicit > info)
- **Filtres** : stance (institutional/investor/research), trade signal (explicit/implicit/none), domaine (7 themes)
- **Divergences inter-docs** : tableau Entity | Bullish Source | Score+ | Bearish Source | Score- | Delta
- **Recherche semantique** : query texte libre -> top 5 passages ChromaDB (cosine similarity)
- **Bouton re-ingest** : relance le parsing si nouveaux PDFs ajoutes

### Tab 4 — Backtest
Validation du pouvoir predictif :
- **Source** : Tweets CSV ou Corpus Documents
- **Asset** : WTI, Brent, Gold, SPX, NASDAQ, EUR/USD
- **Horizon** : 15min a 1 semaine (Alpha Vantage <1h, yfinance >=1h)
- **Resultats** : Directional Accuracy, Pearson/Spearman r, forward returns par bucket, rolling correlation, scatter + OLS trendline

## Re-ingest corpus

```bash
# Ajouter un PDF dans data_corpus/, puis :
python scripts/ingest_corpus.py
# Le script skip automatiquement les fichiers deja traites.

# Pour tout re-ingerer :
python scripts/ingest_corpus.py --force

# Avec FinBERT au lieu de VADER :
python scripts/ingest_corpus.py --model finbert

# Ou depuis le dashboard : Tab 3 -> bouton "Re-ingest corpus"
```

## Variables d'environnement (.env)

```env
TWITTER_BACKEND=csv              # csv | mock | snscrape | api
ALPHA_VANTAGE_KEY=               # requis si horizon < 1h en backtest
SENTIMENT_MODEL=vader            # vader | finbert
DB_PATH=shared/db/sentiment.db
```

## Donnees

| Source | Fichiers | Volume |
|--------|----------|--------|
| Tweets | `data_tweet/financial_juice_tweets.csv` | 607 lignes, 492 exploitables (Mar 28-31 2026) |
| Corpus | `data_corpus/*.pdf` | 14 PDFs macro (Goldman, BofA, Macquarie, Natixis, SEB, etc.) |

## Architecture

```
sentiment_platform/
├── config.py                              # Config centralisee
├── shared/
│   ├── nlp/                               # VADER + FinBERT (singleton)
│   ├── backtest/                          # PriceClient + BacktestEngine
│   └── db/                                # SQLite (6 tables) + ChromaDB (chroma/)
├── module1_twitter/
│   ├── twitter/                           # CSV backend + client factory
│   ├── nlp/                               # Preprocessor tweets
│   └── signal/                            # Signal extractor (log-weighted)
├── module2_nlp/
│   ├── ingestion/                         # Parsers PDF/HTML/TXT
│   ├── nlp/                               # Chunker + NER + Pipeline
│   ├── signal/                            # Aggregator multi-docs
│   └── analysis/                          # BERTopic + Consensus + Cross-source + ChromaDB
│       ├── document_classifier.py         # 7-theme TF-IDF + stance + trade extraction
│       └── summarizer.py                  # Extractive summary (TF-IDF sentence ranking)
├── dashboard/
│   └── app.py                             # Streamlit 4 tabs
├── scripts/
│   └── ingest_corpus.py                   # CLI ingest PDFs -> ChromaDB + SQLite
└── data_tweet/ + data_corpus/             # Donnees reelles
```

## Stack technique

| Couche | Outil |
|--------|-------|
| Sentiment | FinBERT (ProsusAI) + VADER (fallback) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384 dims, 80MB) |
| Topic modeling | BERTopic + KMeans (sklearn) |
| Vector store | ChromaDB (persist_directory) |
| Document classification | TF-IDF keyword scoring, 7 themes, lexical stance/trade markers |
| Consensus/divergence | Marqueurs lexicaux + variance inter-sources + cosine similarity |
| Prix marche | yfinance (>=1h) + Alpha Vantage (<1h) |
| DB metadata | SQLite via SQLAlchemy |
| Dashboard | Streamlit + Plotly |

## Documentation

- `CLAUDE.md` — Spec complete + tracking d'implementation
- `DOCUMENTATION.md` — Signatures API de tous les modules
- `notes_choix_techniques.md` — Arbitrages techniques et justifications
