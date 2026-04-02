# Sentiment Trading Platform — Macro Intelligence

Plateforme d'analyse de sentiment macro combinant flux Twitter et corpus documentaire.
Deux sources : 607 tweets FinancialJuice (données réelles) + 14 PDFs macro (Goldman, BofA, etc.)

## Statut des features

| Feature | Statut |
|---|---|
| Structure monorepo + config | ✅ DONE |
| DB SQLite (5 tables + CRUD) | ✅ DONE |
| Shared NLP — VADER + FinBERT | ✅ DONE |
| Backtest Engine (yfinance + AV) | ✅ DONE |
| Twitter — CSV backend (FinancialJuice) | ✅ DONE |
| Corpus — ChromaDB store (semantic search) | ✅ DONE |
| Corpus — Ingest PDFs (ChromaDB + SQLite) | ✅ DONE |
| Analysis — BERTopic topic modeling | ✅ DONE |
| Analysis — Consensus/Divergence detection | ✅ DONE |
| Analysis — Cross-source alignment | ✅ DONE |
| Dashboard — Tab Macro Digest | ✅ DONE |
| Dashboard — Tab Tweet Intelligence | ✅ DONE |
| Dashboard — Tab Corpus Analysis | ✅ DONE |
| Dashboard — Tab Backtest | ✅ DONE |

## Setup

```bash
git clone <repo>
cd sentiment_platform
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env

# Init DB
python -c "from shared.db.database import init_db; init_db()"

# Ingest corpus PDFs (one-shot, ~2-5 min selon hardware)
python scripts/ingest_corpus.py

# Lancer le dashboard
streamlit run dashboard/app.py
```

## Re-ingest si nouveau PDF ajouté

```bash
# Ajouter le PDF dans data_corpus/, puis :
python scripts/ingest_corpus.py
# Le script skip automatiquement les fichiers déjà traités.
# Ou depuis le dashboard : Tab 3 sidebar → bouton "🔄 Re-ingest corpus"
```

## Variables d'environnement (.env)

```env
TWITTER_BACKEND=csv              # csv | mock | snscrape | api
ALPHA_VANTAGE_KEY=               # requis si horizon < 1h en backtest
SENTIMENT_MODEL=vader            # vader | finbert
DB_PATH=shared/db/sentiment.db
```

## Données

- `data_tweet/financial_juice_tweets.csv` — 607 tweets FinancialJuice (Mar 28–31 2026)
- `data_corpus/*.pdf` — 14 documents macro (Goldman Sachs, BofA, Macquarie, Natixis, SEB…)

## Architecture

Voir `CLAUDE.md` pour la spec complète et `notes_choix_techniques.md` pour les arbitrages.
