# CLAUDE.md — Sentiment Trading Platform

## Contexte & Objectif
Application destinée à une équipe de 9 traders professionnels sans background code.
Deux modules complémentaires dans un monorepo :
- **Module 1 (Twitter)** : scraper des tweets par mots-clés / comptes → signal sentiment sur commodities & equities
- **Module 2 (NLP Docs)** : ingérer des documents financiers (news, hedge fund letters, earnings calls, FOMC) → sentiment par entité

Les deux modules partagent : moteur sentiment (VADER/FinBERT), backtest engine, price client, dashboard Streamlit unifié.

**Stack** : Python 3.11+, Streamlit, SQLite (SQLAlchemy), Plotly, PyTorch

---

## Architecture Monorepo

```
sentiment_platform/
├── CLAUDE.md
├── README.md                          # Mis à jour après chaque feature
├── requirements.txt
├── .env.example
├── config.py                          # Config centralisée unique
├── shared/
│   ├── __init__.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── vader_sentiment.py         # Partagé entre les deux modules
│   │   └── finbert_sentiment.py       # Partagé — singleton, chargé une fois
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── price_client.py            # yfinance (≥1h) + Alpha Vantage (<1h)
│   │   └── engine.py                  # 3 métriques de validation
│   └── db/
│       ├── __init__.py
│       └── database.py                # Init SQLite + 5 tables + toutes les fonctions CRUD
├── module1_twitter/
│   ├── __init__.py
│   ├── twitter/
│   │   ├── __init__.py
│   │   ├── client.py                  # Abstraction layer 3 backends
│   │   ├── mock_backend.py            # ~200 tweets synthétiques, 4 scénarios
│   │   ├── snscrape_backend.py
│   │   └── api_backend.py             # tweepy v2
│   ├── nlp/
│   │   └── preprocessor.py            # Nettoyage spécifique tweets
│   └── signal/
│       └── extractor.py               # Signal pondéré (log followers * retweets * authority)
├── module2_nlp/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base_parser.py             # Classe abstraite BaseParser
│   │   ├── pdf_parser.py              # PyMuPDF
│   │   ├── html_parser.py             # BeautifulSoup + requests
│   │   ├── txt_parser.py
│   │   └── mock_documents.py          # 6 documents synthétiques
│   ├── nlp/
│   │   ├── chunker.py                 # Découpage en chunks (200 mots, overlap 50)
│   │   ├── ner.py                     # spaCy + TRACKED_ENTITIES custom
│   │   └── pipeline.py                # Orchestrateur : parse → chunk → NER → sentiment → DB
│   └── signal/
│       └── aggregator.py              # Agrégation multi-documents par entité
└── dashboard/
    ├── __init__.py
    └── app.py                         # Streamlit — 4 tabs
```

---

## Plan de développement & Orchestration des sub-agents

5 phases. Chaque phase se termine par **tests + review + update README** avant de passer à la suivante.
Les tâches `[PARALLEL]` peuvent être assignées à des sub-agents distincts simultanément.

---

### PHASE 1 — Fondations (séquentiel — tout le reste en dépend)

**Tâches séquentielles :**
1. Créer la structure de dossiers complète avec tous les `__init__.py`
2. Écrire `config.py` (spec complète ci-dessous)
3. Écrire `shared/db/database.py` (5 tables + toutes les fonctions CRUD)
4. Écrire `requirements.txt` et `.env.example`
5. Écrire le `README.md` initial (template ci-dessous)

**✅ TESTS PHASE 1 :**
```bash
python -c "from shared.db.database import init_db; init_db(); print('✅ DB OK')"
python -c "import config; print('✅ Config OK:', list(config.ASSETS.keys()))"
python -c "from shared.db.database import save_tweet, get_tweets; print('✅ CRUD OK')"
```

**✅ REVIEW PHASE 1 :**
- Toutes les tables créées avec les bonnes colonnes et types
- Tous les imports fonctionnent depuis la racine `sentiment_platform/`
- `save_document()` idempotente (ON CONFLICT IGNORE sur SHA256 PK)
- README créé avec statut "⬜ TODO" sur toutes les features
- Mettre à jour README : Phase 1 → ✅ DONE

---

### PHASE 2 — Shared NLP + Backtest Engine

**`[PARALLEL]` Sub-agent A — Shared NLP**

Fichiers : `shared/nlp/vader_sentiment.py`, `shared/nlp/finbert_sentiment.py`

Interface commune obligatoire pour les deux classes :
```python
def analyze(self, text: str) -> dict:
    return {
        "score":      float,   # -1.0 à +1.0
        "label":      str,     # "positive" | "negative" | "neutral"
        "confidence": float,   # 0.0 à 1.0
        "model":      str,     # "vader" | "finbert"
    }
```

`FinBERTSentiment` doit être un **singleton** (chargé une fois au premier import, pas à chaque appel).
Utiliser `ProsusAI/finbert` depuis HuggingFace. Ajouter au lexique VADER les termes financiers :
`bullish(+3), bearish(-3), crash(-3.5), surge(+2.5), ceasefire(+1.5), sanctions(-2),
tightening(-1.5), easing(+1.5), upgrade(+2), downgrade(-2), recession(-3), rally(+2.5)`

**✅ TESTS Sub-agent A :**
```python
from shared.nlp.vader_sentiment import VaderSentiment
from shared.nlp.finbert_sentiment import FinBERTSentiment

vader = VaderSentiment()
finbert = FinBERTSentiment()

cases = [
    ("Oil prices surge as OPEC cuts production sharply", "positive"),
    ("Market crashes on recession fears, investors panic", "negative"),
    ("Trading volumes remain stable this week", "neutral"),
]
for text, expected in cases:
    rv = vader.analyze(text)
    rf = finbert.analyze(text)
    for r, name in [(rv, "VADER"), (rf, "FinBERT")]:
        assert -1 <= r["score"] <= 1, f"{name} score out of range"
        assert r["label"] in ["positive","negative","neutral"]
        assert 0 <= r["confidence"] <= 1
    print(f"[{expected}] VADER={rv['label']}({rv['score']:.2f}) FinBERT={rf['label']}({rf['score']:.2f})")

# Vérifier que FinBERT est bien un singleton (même objet)
fb1 = FinBERTSentiment()
fb2 = FinBERTSentiment()
assert fb1 is fb2, "FinBERT doit être un singleton"
print("✅ NLP tests passed")
```

---

**`[PARALLEL]` Sub-agent B — Backtest Engine + Price Client**

Fichiers : `shared/backtest/price_client.py`, `shared/backtest/engine.py`

`PriceClient.get_prices(asset, start, end, horizon_hours)` :
- `horizon_hours < 1` → Alpha Vantage 1min (nécessite `ALPHA_VANTAGE_KEY`)
- `1 ≤ horizon_hours ≤ 24` → yfinance interval="1h"
- `horizon_hours > 24` → yfinance interval="1d"
- Retourne toujours `pd.DataFrame` avec index `DatetimeIndex UTC` et colonne `"close"`
- `_resolve_ticker()` cherche dans `ASSETS` puis `ENTITY_TICKERS`

`BacktestEngine.run(signals, asset, horizon_hours)` retourne :
```python
{
    "forward_returns_by_bucket": dict,   # 5 buckets Very Negative → Very Positive
    "rolling_correlation":       list,   # [{timestamp, rolling_corr}] fenêtre 20 obs
    "directional_accuracy":      float,  # ratio correct / total
    "accuracy_by_quintile":      dict,   # Q1 (weak) → Q5 (strong)
    "confusion_matrix":          dict,   # {TP, TN, FP, FN}
    "n_signals":                 int,
    "pearson_r":                 float,
    "pearson_pval":              float,
    "spearman_r":                float,
    "raw_df":                    pd.DataFrame,  # pour scatter plot
}
```
Retourner `{"error": "...", "n_signals": N}` si moins de 5 points.

**✅ TESTS Sub-agent B :**
```python
from shared.backtest.price_client import PriceClient
from shared.backtest.engine import BacktestEngine
from datetime import datetime, timedelta
import pandas as pd

pc = PriceClient()
prices = pc.get_prices("WTI", datetime(2024,1,1), datetime(2024,3,1), horizon_hours=24)
assert isinstance(prices, pd.DataFrame)
assert "close" in prices.columns
assert len(prices) > 0
assert prices.index.tz is not None  # UTC timezone
print(f"✅ yfinance: {len(prices)} candles WTI")

engine = BacktestEngine()
signals = [
    {"timestamp": datetime(2024,1,15,tzinfo=__import__('pytz').UTC) + timedelta(days=i),
     "signal_value": 0.7 if i % 2 == 0 else -0.5}
    for i in range(40)
]
results = engine.run(signals, "WTI", horizon_hours=24)
required = ["forward_returns_by_bucket","rolling_correlation","directional_accuracy",
            "accuracy_by_quintile","confusion_matrix","n_signals","pearson_r","spearman_r"]
assert all(k in results for k in required), f"Missing keys: {[k for k in required if k not in results]}"
assert results["n_signals"] >= 5
assert 0 <= results["directional_accuracy"] <= 1
print(f"✅ Backtest: n={results['n_signals']}, acc={results['directional_accuracy']:.2%}, r={results['pearson_r']:.3f}")
```

**✅ REVIEW PHASE 2 :**
- Interfaces identiques entre VaderSentiment et FinBERTSentiment (mêmes clés retournées)
- PriceClient switche correctement selon `horizon_hours`
- BacktestEngine gère le cas `< 5 points` sans crash
- Mettre à jour README : Shared NLP + Backtest → ✅ DONE

---

### PHASE 3 — Module 1 Twitter

**`[PARALLEL]` Sub-agent A — Twitter Backends + Preprocessor**

Fichiers : `module1_twitter/twitter/client.py`, `mock_backend.py`, `snscrape_backend.py`, `api_backend.py`, `module1_twitter/nlp/preprocessor.py`

Format normalisé **obligatoire** pour tous les backends :
```python
{
    "id":              str,
    "created_at":      datetime,   # naive UTC
    "author":          str,
    "followers_count": int,        # jamais None, défaut 0
    "text":            str,
    "retweet_count":   int,        # jamais None, défaut 0
    "like_count":      int,        # jamais None, défaut 0
}
```

`TwitterClient(backend)` : instancie le bon backend selon `"mock"|"snscrape"|"api"`.
Méthodes : `search(keywords, limit)` et `get_user_tweets(username, limit)`.

`mock_backend.py` : ~200 tweets sur 4 scénarios macro, timestamps étalés sur 48h depuis `now()`.
~30% de tweets neutres/bruit dans chaque scénario. Utiliser `faker` pour les comptes non-autorité.

**Scénarios :**
- S1 : Cessez-le-feu Iran/US → WTI baissier (Trump 100M, BBCWorld 20M, Reuters 15M + 15 fake)
- S2 : Coupe OPEC → WTI haussier (zerohedge 1.2M, RaoulGMI 1M + fake energy accounts)
- S3 : Inflation US surprend → SPX baissier, Gold haussier (markets 2M, elerianmohamed 500k)
- S4 : Tensions Russie → Gold spike (various accounts)

`preprocessor.clean_tweet(text)` : lowercase → strip URLs → strip @mentions (garder le texte) → hashtags sans # → normaliser synonymes → strip spéciaux → normaliser espaces.
Synonymes : `"crude oil"→"oil"`, `"petroleum"→"oil"`, `"xau/usd"→"gold"`, `"federal reserve"→"Fed"`.

**✅ TESTS Sub-agent A :**
```python
from module1_twitter.twitter.client import TwitterClient
from module1_twitter.nlp.preprocessor import clean_tweet

client = TwitterClient(backend="mock")
tweets = client.search(["oil","crude"], limit=50)
assert len(tweets) == 50
required = ["id","created_at","author","followers_count","text","retweet_count","like_count"]
assert all(k in tweets[0] for k in required)
assert all(isinstance(t["followers_count"], int) for t in tweets)
assert all(isinstance(t["retweet_count"], int) for t in tweets)
print(f"✅ Mock: {len(tweets)} tweets, format OK")

user_tweets = client.get_user_tweets("realDonaldTrump", limit=10)
assert len(user_tweets) == 10
print("✅ get_user_tweets OK")

raw = "Check out https://t.co/xyz #OPEC @BBCWorld Crude Oil prices CRASH!!! 🔥"
clean = clean_tweet(raw)
assert "https" not in clean
assert "@" not in clean
assert "#" not in clean
assert "oil" in clean.lower()
assert "opec" in clean.lower()
print(f"✅ Preprocessor: '{clean}'")
```

---

**`[PARALLEL]` Sub-agent B — Signal Extractor**

Fichier : `module1_twitter/signal/extractor.py`

```python
def compute_weighted_signal(tweets: list[dict], window_minutes: int, asset: str) -> dict:
    """
    Filtre les tweets par asset_tag et fenêtre temporelle.
    Poids = log(followers+1) * (1 + log(retweets+1)) * authority
    authority = 3.0 si author in HIGH_AUTHORITY_ACCOUNTS, sinon 1.0
    Rationale log() : évite écrasement du signal par un seul compte.
    """
```
Retourne : `{"signal": float, "tweet_count": int, "alert": bool}`

**✅ TESTS Sub-agent B :**
```python
from module1_twitter.signal.extractor import compute_weighted_signal
from datetime import datetime, timedelta

now = datetime.utcnow()
tweets = [
    {"author": "realDonaldTrump", "followers_count": 100_000_000,
     "retweet_count": 50000, "sentiment_score": -0.8,
     "asset_tag": "WTI", "created_at": now - timedelta(minutes=30)},
    {"author": "randomuser123", "followers_count": 500,
     "retweet_count": 2, "sentiment_score": 0.9,
     "asset_tag": "WTI", "created_at": now - timedelta(minutes=10)},
    {"author": "zerohedge", "followers_count": 1_200_000,
     "retweet_count": 1500, "sentiment_score": -0.7,
     "asset_tag": "WTI", "created_at": now - timedelta(minutes=50)},
    # Hors fenêtre → ignoré
    {"author": "markets", "followers_count": 2_000_000,
     "retweet_count": 3000, "sentiment_score": 0.9,
     "asset_tag": "WTI", "created_at": now - timedelta(minutes=90)},
    # Mauvais asset → ignoré
    {"author": "goldtrader", "followers_count": 50000,
     "retweet_count": 100, "sentiment_score": 0.8,
     "asset_tag": "GOLD", "created_at": now - timedelta(minutes=5)},
]
result = compute_weighted_signal(tweets, window_minutes=60, asset="WTI")
assert result["tweet_count"] == 3, f"Expected 3, got {result['tweet_count']}"
assert result["signal"] < 0, f"Trump+zerohedge bearish → signal doit être négatif, got {result['signal']}"
assert "alert" in result
print(f"✅ Signal: {result['signal']:.4f}, count={result['tweet_count']}, alert={result['alert']}")

# Fenêtre vide
empty = compute_weighted_signal([], window_minutes=60, asset="WTI")
assert empty["signal"] == 0.0 and empty["tweet_count"] == 0
print("✅ Edge case empty OK")
```

**✅ REVIEW PHASE 3 :**
- Les 3 backends retournent exactement le même format normalisé
- Pondération HA fonctionne : signal dominé par Trump/zerohedge sur random accounts
- `window_minutes` filtre correctement (tweets hors fenêtre ignorés)
- `asset_tag` filtre correctement (autres assets ignorés)
- Mettre à jour README : Module 1 Twitter → ✅ DONE

---

### PHASE 4 — Module 2 NLP Documents

**`[PARALLEL]` Sub-agent A — Parsers + Mock Documents**

Fichiers : `base_parser.py`, `pdf_parser.py`, `html_parser.py`, `txt_parser.py`, `mock_documents.py`

Format retourné par tous les parsers (même clés obligatoires) :
```python
{
    "title":        str,
    "source":       str,
    "doc_type":     str,       # news | hedge_fund_letter | earnings_call | fomc_minutes | research_note
    "published_at": datetime | None,
    "text":         str,       # texte brut nettoyé, len > 0
}
```

`pdf_parser.py` : utiliser `PyMuPDF (fitz)`. Accepter chemin fichier (str) ou bytes.
`html_parser.py` : BeautifulSoup, supprimer nav/footer/script/style. Chercher `article:published_time` dans les meta tags.
`mock_documents.py` : fonction `get_mock_documents() -> list[dict]` retournant 6 docs.

| # | Source | doc_type | Asset signal | published_at |
|---|---|---|---|---|
| 1 | "Bridgewater Q3 Letter" | hedge_fund_letter | gold+, equities- | 2024-10-15 |
| 2 | "Reuters" | news | oil- (accord Iran) | 2024-09-12 |
| 3 | "Federal Reserve" | fomc_minutes | rates-, SPX- | 2024-11-07 |
| 4 | "ExxonMobil Q3 2024" | earnings_call | oil neutre, China- | 2024-10-25 |
| 5 | "JPMorgan Research" | research_note | gold+ (upgrade OW) | 2024-08-20 |
| 6 | "Bloomberg" | news | oil+ (OPEC cut) | 2024-11-30 |

Chaque document : 300–1000 mots, style réaliste, entités financières clairement mentionnées.

**✅ TESTS Sub-agent A :**
```python
from module2_nlp.ingestion.mock_documents import get_mock_documents
from module2_nlp.ingestion.txt_parser import TxtParser

mocks = get_mock_documents()
assert len(mocks) == 6
required = ["title","source","doc_type","published_at","text"]
for doc in mocks:
    assert all(k in doc for k in required), f"Missing keys in {doc.get('title')}"
    assert len(doc["text"]) >= 300, f"Text too short: {doc.get('title')}"
    assert doc["published_at"] is not None
    assert doc["doc_type"] in ["news","hedge_fund_letter","earnings_call","fomc_minutes","research_note"]
print(f"✅ Mock docs: {len(mocks)} docs, format OK")

parser = TxtParser()
result = parser.parse("Oil prices fell sharply after the ceasefire announcement with Iran.")
assert all(k in result for k in required)
assert len(result["text"]) > 0
print("✅ TxtParser OK")
```

---

**`[PARALLEL]` Sub-agent B — NER + Chunker**

Fichiers : `module2_nlp/nlp/chunker.py`, `module2_nlp/nlp/ner.py`

`chunk_text(text, chunk_size, overlap)` : respecter les limites de phrases (ne pas couper en milieu de phrase). Retourner `list[str]`.

`extract_entities_with_context(text)` : combiner NER spaCy (`en_core_web_sm`) + `TRACKED_ENTITIES` de config. Labels spaCy à conserver : `ORG, PERSON, GPE, PRODUCT, MONEY`. Pour chaque entité, extraire `ENTITY_CONTEXT_WINDOW` tokens de contexte autour. Dédupliquer par `(entity.lower(), entity_type)`.

Retourne `list[dict]` avec clés : `entity`, `entity_type`, `context_text`.

**✅ TESTS Sub-agent B :**
```python
from module2_nlp.nlp.chunker import chunk_text
from module2_nlp.nlp.ner import extract_entities_with_context

# Test chunker
long_text = " ".join(["Oil prices fell sharply. OPEC cut production overnight. Gold surged on safe haven demand."] * 30)
chunks = chunk_text(long_text, chunk_size=200, overlap=50)
assert len(chunks) > 1
for c in chunks:
    assert len(c.split()) <= 260, f"Chunk too large: {len(c.split())} words"
print(f"✅ Chunker: {len(chunks)} chunks")

# Test NER
text = """We remain extremely bullish on crude oil given persistent supply constraints from OPEC.
Meanwhile we are reducing equity exposure as recession fears mount.
The Federal Reserve aggressive tightening is weighing on risk assets.
Gold remains our preferred safe haven in this environment."""
entities = extract_entities_with_context(text)
names = [e["entity"].lower() for e in entities]
assert any("oil" in n or "crude" in n for n in names), f"Oil not found: {names}"
assert any("gold" in n for n in names), f"Gold not found: {names}"
assert all("context_text" in e and len(e["context_text"]) > 10 for e in entities)
# Pas de doublons
seen = [(e["entity"].lower(), e["entity_type"]) for e in entities]
assert len(seen) == len(set(seen)), "Duplicates found"
print(f"✅ NER: {len(entities)} entities: {names}")
```

---

**`[PARALLEL]` Sub-agent C — Pipeline + Aggregator**

Fichiers : `module2_nlp/nlp/pipeline.py`, `module2_nlp/signal/aggregator.py`
Aussi : compléter les fonctions DB `save_document()`, `save_entity_sentiment()`, `save_doc_signal()` dans `shared/db/database.py` si pas encore faites.

`process_document(doc_dict, model)` : reçoit un dict au format BaseParser, retourne `{"document": dict, "signals": dict[entity → signal_dict]}`.
ID du document = SHA256 du texte [:16]. Idempotent (appels multiples sur le même doc ne dupliquent pas).

`aggregate_signals(signals, entity, doc_type_filter)` : pondération par `mention_count`.

**✅ TESTS Sub-agent C :**
```python
from module2_nlp.nlp.pipeline import process_document
from module2_nlp.ingestion.mock_documents import get_mock_documents
from module2_nlp.signal.aggregator import aggregate_signals

mocks = get_mock_documents()

# Doc 2 : Reuters oil bearish
doc_oil = mocks[1]
result = process_document(doc_oil, model="vader")
assert "document" in result and "signals" in result
sig_entities = [k.lower() for k in result["signals"].keys()]
assert any("oil" in e or "crude" in e or "wti" in e for e in sig_entities), \
    f"Oil not detected in: {sig_entities}"
print(f"✅ Pipeline doc2: entities={sig_entities}")

# Idempotence
result2 = process_document(doc_oil, model="vader")
assert result2["document"]["id"] == result["document"]["id"]
print("✅ Idempotence OK")

# Aggregator sur plusieurs docs
all_sigs = []
for doc in mocks:
    r = process_document(doc, model="vader")
    for entity, sig in r["signals"].items():
        all_sigs.append({**sig, "entity": entity, "doc_type": doc["doc_type"]})

agg = aggregate_signals(all_sigs, entity="oil")
assert "signal" in agg and "n_docs" in agg and "total_mentions" in agg
print(f"✅ Aggregator oil: signal={agg['signal']:.4f}, n_docs={agg['n_docs']}")

# Filter par doc_type
agg_news = aggregate_signals(all_sigs, entity="oil", doc_type_filter="news")
assert agg_news["n_docs"] <= agg["n_docs"]
print(f"✅ Aggregator filter news: n_docs={agg_news['n_docs']}")
```

**✅ REVIEW PHASE 4 :**
- Tous les parsers retournent exactement le même format (mêmes clés)
- NER : pas de doublons, contexte non vide, types valides
- Pipeline : idempotent sur SHA256, gère chunks > 512 tokens via truncation
- Aggregator : pondération mention_count correcte
- Mettre à jour README : Module 2 NLP → ✅ DONE

---

### PHASE 5 — Dashboard Streamlit Unifié (séquentiel, dépend de tout)

Fichier : `dashboard/app.py`

**Règles absolues pour le dashboard :**
- Jamais de `print()` → utiliser `st.error()`, `st.warning()`, `st.info()`
- Tous les appels réseau dans des blocs `try/except` avec `st.error()` gracieux
- `st.spinner()` sur toutes les opérations longues (FinBERT, yfinance, scraping)
- Les traders n'utilisent pas le terminal : l'app doit être 100% autonome

**Structure : 4 tabs**

#### 🐦 Tab 1 — Twitter Live Signal

Sidebar :
- `st.radio` Modèle : VADER / FinBERT
- `st.radio` Backend : Mock / snscrape / API
- `st.selectbox` Asset : WTI / Brent / Gold / SPX
- `st.multiselect` Keywords (pré-remplis depuis config)
- `st.text_input` Compte custom à suivre
- `st.button("🔄 Refresh")` → scraping + analyse + sauvegarde DB

Contenu :
1. **3 gauges Plotly** en colonnes : signal 1h / 4h / 24h
   - `plotly.graph_objects.Indicator(mode="gauge+number")`
   - Rouge si < -0.4, vert si > 0.4, gris sinon
   - Sous chaque gauge : label "⬇ BEARISH WTI" / "→ NEUTRAL" / "⬆ BULLISH WTI"
2. **Line chart** `px.line` double axe Y : signal_value (gauche) + prix yfinance (droite) sur 7j
   - Marqueurs sur les alertes (rouge = négatif, vert = positif)
3. **Dataframe** tweets : colonnes timestamp | author | tweet | score | label
   - Color-coding via `st.dataframe(df.style.applymap(color_label, subset=["label"]))`
4. **2 colonnes** : pie chart pos/neg/neutral + bar chart top 10 mots

#### 📊 Tab 2 — Twitter Backtest

Contrôles :
- `st.date_input` start / end
- `st.select_slider` Horizon : "15min","30min","1h","4h","12h","1d","3d","1w"
  - `st.info` auto : "📡 Source : Alpha Vantage 1min" si <1h, sinon "📈 Source : yfinance (1h/1d)"
- `st.button("▶ Run Backtest")`

Résultats (5 graphiques Plotly) :
1. KPIs en 5 colonnes : Directional Accuracy | Pearson r | Spearman r | N signals | p-value
2. Bar chart : Forward Returns by Bucket (rouge si négatif, vert si positif)
3. Line chart : Rolling Correlation sur 20 obs (ligne pointillée à y=0)
4. Heatmap `px.imshow` : Confusion Matrix avec annotations TP/TN/FP/FN
5. Scatter `px.scatter` : signal vs return + OLS trendline (`trendline="ols"`)
6. Bar chart : Accuracy by Quintile (Q1 → Q5)

#### 📄 Tab 3 — Analyser un Document

Sidebar :
- `st.radio` Modèle : VADER / FinBERT
- `st.selectbox` Type : news / hedge_fund_letter / earnings_call / fomc_minutes
- `st.radio` Mode entrée : 📎 Upload PDF | 🔗 URL | ✏️ Texte libre | 📚 Document mock

Contenu selon mode :
- Upload : `st.file_uploader(type=["pdf"])`
- URL : `st.text_input` + bouton "Fetch"
- Texte : `st.text_area`
- Mock : `st.selectbox` parmi les 6 mocks (afficher title + source)

`st.button("🔍 Analyser")` → `pipeline.process_document()` avec spinner

Résultats en 2 colonnes :
- Colonne gauche :
  - `st.dataframe` heatmap entités : entity | score | mentions | label (color-coded)
  - `st.metric` pour le top 3 alertes (\|signal\| le plus fort)
- Colonne droite :
  - `px.bar` horizontal : entity (Y) vs score (X), couleurs rouge/vert
  - Texte annoté : `st.markdown` avec entités surlignées via `<span style="background:...">` inline HTML

#### 📈 Tab 4 — Document Backtest

Contrôles :
- `st.selectbox` Entité (liste depuis DB `get_all_entities()`)
- `st.selectbox` Filter doc_type (tous + chaque type)
- `st.date_input` start / end
- `st.select_slider` Horizon (même que Tab 2 + même `st.info` source)
- `st.selectbox` Asset prix à corréler (oil/gold/SPX/NASDAQ/EUR-USD)
- `st.button("▶ Run Backtest")`

Résultats : identiques à Tab 2

---

**✅ TESTS PHASE 5 :**
```bash
# Smoke test : dashboard démarre sans erreur
streamlit run dashboard/app.py --server.headless true &
sleep 8
curl -sf http://localhost:8501/_stcore/health && echo "✅ Dashboard UP" || echo "❌ Dashboard FAILED"
kill %1 2>/dev/null
```

Tests manuels à cocher :
- [ ] Tab 1 : Refresh mock → gauges peuplées, pas d'erreur
- [ ] Tab 1 : Switch VADER → FinBERT → scores changent
- [ ] Tab 1 : Switch backend Mock → snscrape → aucun crash
- [ ] Tab 2 : Run backtest WTI, horizon 1d → 5 graphiques affichés
- [ ] Tab 3 : Charger doc mock "Reuters oil" → oil détecté, score négatif
- [ ] Tab 3 : Switch VADER → FinBERT → scores mis à jour
- [ ] Tab 4 : Run backtest entité "oil", horizon 1d → résultats cohérents
- [ ] Aucun `print()` dans les logs Streamlit

**✅ REVIEW PHASE 5 :**
- Dashboard 100% utilisable sans terminal par un non-dev
- Toutes les erreurs catchées avec messages clairs (`st.error`)
- Labels en anglais (audience internationale)
- Mettre à jour README : toutes features → ✅ DONE

---

## Spec complète config.py

```python
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
```

---

## Spec shared/db/database.py — 5 tables + CRUD

Tables :

**`tweets`** : id(TEXT PK), created_at(DATETIME), author(TEXT), followers_count(INTEGER), text(TEXT), text_clean(TEXT), retweet_count(INTEGER), like_count(INTEGER), asset_tag(TEXT), sentiment_score(FLOAT), sentiment_label(TEXT), model_used(TEXT), inserted_at(DATETIME DEFAULT now)

**`tweet_signals`** : id(INTEGER PK AUTOINCREMENT), timestamp(DATETIME), asset(TEXT), window_minutes(INTEGER), signal_value(FLOAT), tweet_count(INTEGER), alert(BOOLEAN)

**`documents`** : id(TEXT PK), title(TEXT), source(TEXT), doc_type(TEXT), published_at(DATETIME), inserted_at(DATETIME DEFAULT now), char_count(INTEGER)

**`entity_sentiments`** : id(INTEGER PK AUTOINCREMENT), document_id(TEXT FK→documents.id), entity(TEXT), entity_type(TEXT), context_text(TEXT), sentiment_score(FLOAT), sentiment_label(TEXT), model_used(TEXT), confidence(FLOAT)

**`document_signals`** : id(INTEGER PK AUTOINCREMENT), document_id(TEXT FK), entity(TEXT), signal_value(FLOAT), mention_count(INTEGER), alert(BOOLEAN), published_at(DATETIME)

Fonctions à exposer :
- `init_db()` → créer toutes les tables si inexistantes
- `save_tweet(tweet_dict)` → INSERT OR IGNORE
- `save_tweet_signal(signal_dict)`
- `save_document(doc_dict)` → INSERT OR IGNORE (idempotent sur PK)
- `save_entity_sentiment(record_dict)`
- `save_doc_signal(signal_dict)`
- `get_tweets(asset, since_minutes)` → list[dict]
- `get_tweet_signals(asset)` → list[dict]
- `get_doc_signals(entity, doc_type_filter=None)` → list[dict]
- `get_all_entities()` → list[str] (pour le selectbox Tab 4)

---

## README.md — Template

```markdown
# Sentiment Trading Platform

Plateforme d'analyse de sentiment financier pour traders.
Deux modules : Twitter scraping + NLP documents → signaux actionnables + backtest.

## Statut des features

| Feature | Statut |
|---|---|
| Structure monorepo + config | ⬜ TODO |
| DB SQLite (5 tables + CRUD) | ⬜ TODO |
| Shared NLP — VADER | ⬜ TODO |
| Shared NLP — FinBERT (singleton) | ⬜ TODO |
| Backtest Price Client (yfinance + AV) | ⬜ TODO |
| Backtest Engine (3 métriques) | ⬜ TODO |
| Twitter — Mock backend | ⬜ TODO |
| Twitter — snscrape backend | ⬜ TODO |
| Twitter — API backend (tweepy) | ⬜ TODO |
| Twitter — Preprocessor | ⬜ TODO |
| Twitter — Signal Extractor | ⬜ TODO |
| NLP Docs — Parsers (PDF/HTML/TXT) | ⬜ TODO |
| NLP Docs — Mock Documents (x6) | ⬜ TODO |
| NLP Docs — Chunker | ⬜ TODO |
| NLP Docs — NER (spaCy + custom) | ⬜ TODO |
| NLP Docs — Pipeline | ⬜ TODO |
| NLP Docs — Signal Aggregator | ⬜ TODO |
| Dashboard — Tab Twitter Live | ⬜ TODO |
| Dashboard — Tab Twitter Backtest | ⬜ TODO |
| Dashboard — Tab Document Analysis | ⬜ TODO |
| Dashboard — Tab Document Backtest | ⬜ TODO |

## Setup

\`\`\`bash
git clone <repo>
cd sentiment_platform
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env        # remplir les clés si nécessaire
python -c "from shared.db.database import init_db; init_db()"
streamlit run dashboard/app.py
\`\`\`

## Variables d'environnement (.env)

\`\`\`env
TWITTER_BACKEND=mock             # mock | snscrape | api
TWITTER_BEARER_TOKEN=            # requis si backend=api
ALPHA_VANTAGE_KEY=               # requis si horizon < 1h en backtest
SENTIMENT_MODEL=vader            # vader | finbert
DB_PATH=shared/db/sentiment.db
\`\`\`

## Architecture

Voir CLAUDE.md pour la spec complète.
```

---

## Règles générales pour Claude Code

1. **Tester après chaque phase** avant de continuer. Ne pas passer à la phase suivante si un test échoue.
2. **Mettre à jour README.md** après chaque feature : `⬜ TODO` → `🔄 IN PROGRESS` → `✅ DONE`.
3. **Jamais de `print()`** dans le dashboard — `st.error()` / `st.warning()` / `st.info()` uniquement.
4. **Tous les imports** depuis la racine `sentiment_platform/` (ajouter au `sys.path` si nécessaire via `__init__.py`).
5. **Gestion d'erreurs** sur tous les appels réseau (Twitter, yfinance, Alpha Vantage) avec fallback gracieux.
6. **FinBERT singleton** : utiliser un pattern module-level `_instance = None` dans `finbert_sentiment.py`.
7. **DB idempotente** : `INSERT OR IGNORE` sur tous les `save_*` pour les tables avec PK métier.
8. **Config via .env** : jamais de clés API hardcodées.
9. **Phases parallèles** : les sub-agents travaillent sur des fichiers disjoints, pas de conflits possibles.

---

## Anticipations présentation orale

**Module 1 — Twitter**
- "Pourquoi pas l'API officielle ?" → 5000$/mois pour Pro, backend prêt plug-and-play le jour J
- "Latence ?" → API Pro streaming quasi temps réel. Snscrape : polling 5min
- "Pourquoi log(followers) ?" → Évite que Trump (100M) écrase seul tout le signal. Compression logarithmique
- "Comment étendre ?" → Reddit PRAW, Telegram channels, RSS Bloomberg — même interface TwitterClient

**Module 2 — Documents**
- "Pourquoi sentiment par entité et pas global ?" → Un doc peut être bullish gold ET bearish equities simultanément. Sentiment global = information perdue
- "FinBERT vs GPT-4 ?" → FinBERT local, gratuit, déterministe, 512 tokens. GPT-4 meilleur sur longs docs mais coût API + latence. Extension naturelle à mentionner
- "Comment gérer les 512 tokens FinBERT ?" → Chunking + sentiment sur contexte autour de l'entité, pas le doc entier
- "Horizon long pour hedge fund letters ?" → Le positioning se déploie sur semaines, horizon 1 semaine–1 mois plus pertinent

**Backtest**
- "p-value pas significative ?" → Normal sur données mock synthétiques. Sur vraies données historiques c'est là que ça devient testable
- "Directional accuracy ~50% ?" → Baseline aléatoire. Sur données réelles avec volume suffisant on s'attend à battre ça
- "Pourquoi Spearman en plus de Pearson ?" → Capture les relations monotones non-linéaires, robuste aux outliers de signal

**Architecture**
- "Pourquoi monorepo ?" → Code partagé (backtest, NLP, DB) sans duplication. Extensible facilement

---

## Repository GitHub
**URL :** https://github.com/tsg1306/MacroSentiment-Analysis

---

## 🔄 SUIVI D'IMPLÉMENTATION — Redesign 2026-04-02
> **Reprendre ici si la session est coupée.** Plan complet : `docs/superpowers/plans/2026-04-02-macro-intel-implementation.md`
> Spec complète : `docs/superpowers/specs/2026-04-02-macro-intel-redesign.md`
> Notes choix techniques : `notes_choix_techniques.md`

### État des tâches

| # | Tâche | Fichiers clés | Statut |
|---|-------|--------------|--------|
| 0 | Livrables texte (notes, CLAUDE.md, README, DOCUMENTATION) | `notes_choix_techniques.md`, `README.md`, `DOCUMENTATION.md` | ✅ DONE |
| 1 | Dépendances | `requirements.txt` | ✅ DONE |
| 2 | DB — table `corpus_ingested` | `shared/db/database.py` | ✅ DONE |
| 3 | CSV backend | `module1_twitter/twitter/csv_backend.py`, `client.py` | ✅ DONE |
| 4 | Corpus Store (ChromaDB) | `module2_nlp/analysis/corpus_store.py` | ✅ DONE |
| 5 | Script ingest PDFs | `scripts/ingest_corpus.py` | ✅ DONE |
| 6 | BERTopic wrapper | `module2_nlp/analysis/topic_model.py` | ✅ DONE |
| 7 | Consensus/Divergence | `module2_nlp/analysis/consensus.py` | ✅ DONE |
| 8 | Cross-source alignment | `module2_nlp/analysis/cross_source.py` | ✅ DONE |
| 9 | Dashboard Tab 1 — Macro Digest | `dashboard/app.py` | ✅ DONE |
| 10 | Dashboard Tab 2 — Tweet Intelligence | `dashboard/app.py` | ✅ DONE |
| 11 | Dashboard Tab 3 — Corpus Analysis | `dashboard/app.py` | ✅ DONE |
| 12 | Dashboard Tab 4 — Backtest | `dashboard/app.py` | ✅ DONE |
| 13 | README + DOCUMENTATION finaux | `README.md`, `DOCUMENTATION.md` | ✅ DONE |

### Légende statuts
- ⬜ TODO — pas encore commencé
- 🔄 IN PROGRESS — en cours
- ✅ DONE — terminé et testé
- ❌ FAILED — erreur bloquante (voir notes inline)

### Règle de mise à jour
**Après chaque tâche terminée**, mettre à jour le statut ci-dessus de ⬜/🔄 → ✅ DONE.
Si une tâche échoue, noter ❌ et ajouter une ligne "Erreur:" avec le message d'erreur.

### Contexte données réelles
- `data_tweet/financial_juice_tweets.csv` : ~492 tweets exploitables (607 lignes brutes dont ~110 suites de lignes multi-line + ~5 URLs filtrées), colonnes: date, author_name, content
- `data_corpus/*.pdf` : 14 PDFs macro (Goldman Sachs, BofA, Macquarie, Natixis, SEB, Alexander Campbell, Cavendish, DBS, Richard Bexelius, Michael Howell, Canaccord x2, Unknown)
- ChromaDB persisté dans : `shared/db/chroma/`
- SQLite : `shared/db/sentiment.db`

---

### ⚠️ Modifications & Fonctionnalités Annulées

| Fonctionnalité | Statut | Raison |
|---------------|--------|--------|
| HDBSCAN clustering (BERTopic) | ❌ Remplacé par KMeans | Pas de wheel précompilé pour Python 3.14/Windows. BERTopic utilise `KMeans(n_clusters=8)` de sklearn à la place. Impact : topics moins adaptatifs (N fixe) mais fonctionnel. Extension : installer MSVC Build Tools + `pip install hdbscan` pour revenir à HDBSCAN. |
| Ollama / LLM local Q&A | ⏳ Non implémenté (prévu TODO) | Hors scope 6h. Architecture ChromaDB prête — ajouter `langchain + ollama` suffit. |
| Signal extractor pondéré (log followers) | 🔄 Désactivé pour CSV backend | Tous les tweets FinancialJuice = même auteur, followers_count=0. Pondération inutile. Le signal est calculé comme moyenne simple des scores de sentiment. |
| Twitter mock_backend | 🔄 Non exposé dans nouveau dashboard | Conservé dans le code mais le backend "mock" n'est plus affiché dans l'UI. Remplacé par CSV réel. |
| mock_documents.py (6 docs synthétiques) | 🔄 Non utilisé | Remplacé par les 14 vrais PDFs dans data_corpus/. Fichier conservé pour compatibilité. |
| Tabs Twitter Live + Twitter Backtest (anciens) | 🔄 Restructurés | Fusionnés dans nouveaux tabs Macro Digest + Tweet Intelligence + Backtest unique. |
- "Scalabilité ?" → Remplacer SQLite par PostgreSQL en changeant la connection string. Ajouter un job scheduler (APScheduler) pour le scraping automatique toutes les 5min
