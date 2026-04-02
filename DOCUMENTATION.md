# Documentation — Sentiment Trading Platform

## 1. `dashboard/`

**Fichier principal : `app.py`** — Dashboard Streamlit avec 4 onglets.

| Fonction | Signature | Description |
|---|---|---|
| `get_vader()` | `() -> VaderSentiment` | Charge et cache l'analyseur VADER (via `@st.cache_resource`) |
| `get_finbert()` | `() -> FinBERTSentiment` | Charge et cache l'analyseur FinBERT (via `@st.cache_resource`) |
| `get_analyzer(model_name)` | `(str) -> VaderSentiment \| FinBERTSentiment` | Retourne le bon analyseur selon "FinBERT" ou "VADER" |
| `color_label(val)` | `(str) -> str` | CSS vert/rouge/neutre pour styliser les labels sentiment dans les DataFrames |
| `color_score(val)` | `(str) -> str` | Idem, utilisee dans Tab 3 |

**4 Onglets :**

- **Tab 1 — Twitter Live Signal** : 3 gauges Plotly (1h/4h/24h), line chart signal vs prix, dataframe tweets, pie chart + bar chart mots
- **Tab 2 — Twitter Backtest** : KPIs, forward returns par bucket, rolling correlation, confusion matrix, scatter signal vs return, accuracy par quintile
- **Tab 3 — Document Analyzer** : Upload PDF / URL / texte / mock -> heatmap entites, top 3 alertes, bar chart entites, texte annote
- **Tab 4 — Document Backtest** : Identique a Tab 2 mais sur les signaux documents

---

## 2. `tests/`

**6 fichiers, 89 tests au total :**

| Fichier | Nb tests | Ce qu'il couvre |
|---|---|---|
| `test_dashboard_smoke.py` | 3 | Imports, config, parsing de app.py |
| `test_shared_nlp.py` | 17 | VADER (score, lexique financier) + FinBERT (singleton, score range, labels) |
| `test_shared_backtest.py` | 14 | Resolution tickers, edge cases (0 ou <5 signaux), cles de sortie, correlations, confusion matrix |
| `test_twitter_pipeline.py` | 18 | Format tweets mock, limites search, preprocessor (URLs, mentions, synonymes, idempotence), ApiBackend validation |
| `test_signal_extractor.py` | 5 | Signal pondere, filtrage asset/fenetre, `compute_all_windows` |
| `test_nlp_pipeline.py` | 18 | Mock docs (6 docs, format), TxtParser, chunker, NER (entites, doublons, contexte), pipeline (structure, idempotence), aggregator (filtres, ponderation) |

---

## 3. `shared/backtest/`

### `price_client.py` — Classe `PriceClient`

| Methode | Signature | Description |
|---|---|---|
| `get_prices()` | `(asset, start, end, horizon_hours) -> pd.DataFrame` | Routage automatique : <1h -> Alpha Vantage 1min, 1-24h -> yfinance 1h, >24h -> yfinance 1d. Retourne DataFrame avec index `DatetimeIndex UTC` et colonne `close` |
| `_resolve_ticker()` | `(asset) -> dict` | Cherche dans `ASSETS` puis `ENTITY_TICKERS`. Raise `KeyError` si inconnu |
| `_fetch_yfinance()` | `(ticker, start, end, interval) -> DataFrame` | Appel yfinance, gere MultiIndex et timezone |
| `_fetch_alpha_vantage()` | `(symbol, start, end) -> DataFrame` | Appel API Alpha Vantage intraday 1min |

### `engine.py` — Classe `BacktestEngine`

| Methode | Signature | Description |
|---|---|---|
| `run()` | `(signals, asset, horizon_hours=24) -> dict` | Backtest complet. Retourne `forward_returns_by_bucket`, `rolling_correlation`, `directional_accuracy`, `accuracy_by_quintile`, `confusion_matrix`, `pearson_r`, `spearman_r`, `raw_df`. Retourne `{"error": ...}` si <5 signaux |
| `_closest_price()` | `(prices, target_time) -> float \| None` | Prix le plus proche d'un timestamp |
| `_bucket_returns()` | `(df) -> dict` | Forward returns moyens par bucket (Very Negative -> Very Positive) |
| `_rolling_correlation()` | `(df, window=20) -> list` | Correlation glissante signal/return |
| `_directional_accuracy()` | `(signals, returns) -> float` | % de fois ou sign(signal) == sign(return) |
| `_accuracy_by_quintile()` | `(df) -> dict` | Accuracy par quintile Q1-Q5 de force du signal |
| `_confusion_matrix()` | `(signals, returns) -> dict` | `{TP, TN, FP, FN}` |

---

## 4. `shared/nlp/`

### `vader_sentiment.py` — Classe `VaderSentiment`

| Methode | Description |
|---|---|
| `__init__()` | Initialise VADER + enrichit le lexique avec 12 termes financiers (bullish +3, crash -3.5, etc.) |
| `analyze(text) -> dict` | Retourne `{score, label, confidence, model:"vader"}`. Seuils : >0.05 = positive, <-0.05 = negative |

### `finbert_sentiment.py` — Classe `FinBERTSentiment` (singleton)

| Methode | Description |
|---|---|
| `__new__(cls)` | Pattern singleton. Charge `ProsusAI/finbert` une seule fois. Fallback VADER si torch/transformers absent |
| `analyze(text) -> dict` | Retourne `{score, label, confidence, model:"finbert"}`. Score = P(positive) - P(negative). Tokenization tronquee a 512 tokens |

**Interface commune** : les deux classes retournent exactement `{score: float[-1,1], label: str, confidence: float[0,1], model: str}`

---

## 5. `shared/db/`

### `database.py` — SQLAlchemy ORM + CRUD

**6 tables ORM :**

| Table | PK | Colonnes cles |
|---|---|---|
| `tweets` | `id` (TEXT) | author, followers_count, text, text_clean, asset_tag, sentiment_score, sentiment_label, model_used |
| `tweet_signals` | `id` (AUTO) | timestamp, asset, window_minutes, signal_value, tweet_count, alert |
| `documents` | `id` (TEXT, SHA256) | title, source, doc_type, published_at, char_count |
| `entity_sentiments` | `id` (AUTO) | document_id (FK), entity, entity_type, context_text, sentiment_score, confidence |
| `document_signals` | `id` (AUTO) | document_id (FK), entity, signal_value, mention_count, alert |
| `corpus_ingested` | `filename` (TEXT) | ingested_at — trace les PDFs deja traites par ingest_corpus.py |

**Fonctions CRUD :**

| Fonction | Description |
|---|---|
| `init_db()` | Cree toutes les tables si inexistantes |
| `get_session()` | Retourne une session SQLAlchemy |
| `save_tweet(dict)` | INSERT OR IGNORE (idempotent sur PK) |
| `save_tweet_signal(dict)` | Insert signal agrege Twitter |
| `save_document(dict)` | INSERT OR IGNORE (idempotent sur SHA256) |
| `save_entity_sentiment(dict)` | Insert sentiment par entite |
| `save_doc_signal(dict)` | Insert signal document |
| `get_tweets(asset?, since_minutes?)` | Filtre par asset et/ou fenetre temporelle |
| `get_tweet_signals(asset?)` | Signaux Twitter tries par timestamp |
| `get_doc_signals(entity?, doc_type_filter?)` | Signaux documents filtres |
| `get_all_entities()` | Liste unique des entites (pour selectbox Tab 4) |
| `get_ingested_files()` | Retourne `set` des filenames deja traites par ingest_corpus.py |
| `mark_file_ingested(filename)` | Marque un PDF comme ingere (idempotent sur PK) |

---

## 6. `module1_twitter/twitter/`

### `client.py` — Classe `TwitterClient` (factory pattern)

| Methode | Description |
|---|---|
| `__init__(backend="mock")` | Instancie le bon backend : `MockBackend`, `SnscrapeBackend`, ou `ApiBackend` |
| `search(keywords, limit=100)` | Delegue la recherche au backend |
| `get_user_tweets(username, limit=10)` | Delegue la recuperation par compte |

### `mock_backend.py` — Classe `MockBackend`

~200 tweets synthetiques sur 4 scenarios macro (~30% bruit chacun) :

- **S1** : Cessez-le-feu Iran/US -> WTI baissier (Trump 100M, BBCWorld, Reuters)
- **S2** : Coupe OPEC -> WTI haussier (zerohedge, RaoulGMI)
- **S3** : Inflation US -> SPX baissier, Gold haussier
- **S4** : Tensions Russie -> Gold spike

| Fonction/Methode | Description |
|---|---|
| `_fake_account()` | Genere un username/followers aleatoire via Faker |
| `_rand_ts(base, spread_hours=48)` | Timestamp aleatoire dans les 48h precedentes |
| `_make_tweet(author, followers, text, base_ts)` | Construit un dict tweet normalise |
| `_build_scenario(signal_texts, noise_texts, authority_map, base_ts, n_total=50)` | Genere 50 tweets par scenario |
| `search(keywords, limit)` | Filtre par mot-cle (case-insensitive) |
| `get_user_tweets(username, limit)` | Filtre par auteur |

### `api_backend.py` / `snscrape_backend.py`

Stubs qui raise `NotImplementedError`. `ApiBackend` valide que `TWITTER_BEARER_TOKEN` est defini.

---

## 7. `module1_twitter/nlp/`

### `preprocessor.py`

| Fonction | Description |
|---|---|
| `clean_tweet(text) -> str` | Pipeline idempotent : lowercase -> strip URLs -> strip @mentions -> hashtags sans # -> synonymes (crude oil->oil, petroleum->oil, xau/usd->gold, federal reserve->Fed) -> strip speciaux -> normalise espaces |

---

## 8. `module1_twitter/signal/`

### `extractor.py`

| Fonction | Description |
|---|---|
| `compute_weighted_signal(tweets, window_minutes, asset) -> dict` | Filtre par asset_tag + fenetre temps. Poids = `log(followers+1) * (1+log(retweets+1)) * authority` (3.0 si HIGH_AUTHORITY, sinon 1.0). Retourne `{signal, tweet_count, alert}` |
| `compute_all_windows(tweets, asset) -> dict` | Appelle `compute_weighted_signal` pour chaque fenetre (60/240/1440 min). Retourne `{"60min": {...}, "240min": {...}, "1440min": {...}}` |

---

## 9. `module2_nlp/ingestion/`

### `base_parser.py` — Classe abstraite `BaseParser`

Methode abstraite `parse(source, **kwargs) -> dict` avec format obligatoire : `{title, source, doc_type, published_at, text}`

### Parsers concrets

| Classe | Fichier | Specificite |
|---|---|---|
| `PdfParser` | `pdf_parser.py` | PyMuPDF (fitz). Accepte chemin fichier ou bytes |
| `HtmlParser` | `html_parser.py` | BeautifulSoup. Supprime nav/footer/script/style. Extrait `article:published_time` |
| `TxtParser` | `txt_parser.py` | Texte brut. Accepte str ou bytes (UTF-8) |

### `mock_documents.py`

| Fonction | Description |
|---|---|
| `get_mock_documents() -> list[dict]` | 6 documents financiers synthetiques (Bridgewater letter, Reuters, FOMC, ExxonMobil earnings, JPMorgan research, Bloomberg OPEC). 300-1000 mots chacun |

---

## 10. `module2_nlp/nlp/`

### `chunker.py`

| Fonction | Description |
|---|---|
| `chunk_text(text, chunk_size=200, overlap=50) -> list[str]` | Decoupe en chunks respectant les limites de phrases. Overlap configurable |

### `ner.py`

| Fonction | Description |
|---|---|
| `extract_entities_with_context(text) -> list[dict]` | Combine spaCy NER (ORG, PERSON, GPE, PRODUCT, MONEY) + `TRACKED_ENTITIES` custom. Deduplique par `(entity.lower(), type)`. Retourne `{entity, entity_type, context_text}` |
| `_find_token_index(tokens, entity_str) -> int` | Recherche floue de l'index d'une entite dans les tokens |
| `_get_context(tokens, start, end) -> str` | Extrait fenetre de contexte (ENTITY_CONTEXT_WINDOW/2 avant/apres) |

### `pipeline.py`

| Fonction | Description |
|---|---|
| `process_document(text, file_type, doc_type, model, title, source, published_at) -> dict` | Pipeline complet : SHA256 doc_id -> init DB -> save document -> chunk -> NER -> sentiment par chunk -> save entity_sentiments -> agrege par entite -> save doc_signals. Idempotent. Retourne `{document: dict, signals: {entity -> signal_dict}}` |

---

## 11. `module2_nlp/signal/`

### `aggregator.py`

| Fonction | Description |
|---|---|
| `aggregate_signals(signals, entity?, doc_type_filter?) -> dict` | Moyenne ponderee par `mention_count` : `signal = sum(signal_value * mentions) / sum(mentions)`. Filtre optionnel par entite (case-insensitive) et doc_type. Retourne `{signal, n_docs, total_mentions}` |

---

## 12. `module2_nlp/db/`

### `database.py`

Simple re-export des fonctions de `shared.db.database` : `init_db`, `save_document`, `save_entity_sentiment`, `save_doc_signal`, `get_doc_signals`, `get_all_entities`. Raccourci d'import pour le module 2.

---

## 13. `module2_nlp/analysis/`

### `corpus_store.py` — ChromaDB wrapper

Stockage et recherche semantique des chunks corpus. Persist directory : `shared/db/chroma/`. Embeddings : `all-MiniLM-L6-v2` (384 dims).

| Fonction | Description |
|---|---|
| `_get_embed_model()` | Singleton SentenceTransformer('all-MiniLM-L6-v2') |
| `_get_collection()` | Retourne collection ChromaDB `corpus_chunks` (cosine space), creee si inexistante |
| `ingest_chunks(chunks, metadatas, doc_id) -> int` | Embed + upsert dans ChromaDB. IDs : `{doc_id}_chunk_{i}`. Batch 100. Idempotent |
| `semantic_search(query, n_results=5, where=None) -> list[dict]` | Recherche semantique. Retourne `{text, source, doc_type, similarity, doc_id}`. Support filtre metadata |
| `get_corpus_theme_embedding(keywords) -> np.ndarray\|None` | Embedding moyen des chunks matchant les keywords (top 20, cosine dist < 0.8). Utilise par cross_source.py |
| `get_collection_count() -> int` | Nombre de chunks en collection |
| `delete_collection()` | Supprime la collection (pour --force re-ingest) |

### `topic_model.py` — BERTopic wrapper

| Fonction | Description |
|---|---|
| `fit_tweet_topics(texts, n_topics=8) -> tuple` | Fit BERTopic avec KMeans (HDBSCAN indisponible). Retourne `(topic_model, topics_list, topic_labels_dict)` |
| `get_topic_barchart(topic_model, top_n=8) -> Figure\|None` | Plotly barchart des top keywords par topic |
| `get_topic_map(topic_model) -> Figure\|None` | Plotly scatter 2D des topics |

### `consensus.py` — Consensus/Divergence detection

| Fonction | Description |
|---|---|
| `classify_tweet(text) -> dict` | Classification par marqueurs lexicaux. Retourne `{label, consensus_score, divergence_score, signal_score}`. Labels : consensus/divergence/signal_faible/neutral |
| `detect_divergences(entity_signals, threshold=0.4) -> list[dict]` | Divergences inter-sources (delta score > threshold). Retourne `{entity, doc_a, score_a, doc_b, score_b, delta}` trie par delta desc |
| `detect_consensus(entity_signals, min_sources=2, max_std=0.20, min_abs_mean=0.25) -> list[dict]` | Consensus inter-sources (faible std, signal fort). Retourne `{entity, mean_score, std, n_sources, direction}` |

### `cross_source.py` — Alignement tweets vs corpus

| Fonction | Description |
|---|---|
| `compute_theme_alignment(keywords, tweet_texts, corpus_theme_emb) -> float\|None` | Cosine similarity entre mean(tweet embeddings) et corpus embedding pour un theme. Retourne [0,1] ou None si < 3 tweets matchent |
| `get_all_theme_alignments(tweet_texts) -> dict` | Alignement pour les 7 themes predefinis (Oil, Gold, Fed, Geopolitics, Equities, China, ECB). Utilise ChromaDB via corpus_store |

---

## 14. `scripts/`

### `ingest_corpus.py` — CLI ingest PDFs

Usage : `python scripts/ingest_corpus.py [--force] [--model vader|finbert]`

| Fonction | Description |
|---|---|
| `guess_doc_type(filename) -> str` | Heuristique filename -> doc_type (research_note, news). Patterns : Weekly/Wrap -> research_note, inflation/war/gold -> news |
| `clear_corpus_ingested_table()` | Vide la table corpus_ingested (utilise avec --force) |
| `ingest(force, model)` | Pipeline principal : pour chaque PDF dans data_corpus/, parse -> chunk -> ChromaDB (ingest_chunks) -> NLP pipeline (NER+sentiment -> SQLite) -> mark_file_ingested. Idempotent sans --force |

**Sortie type :** 14 PDFs, ~630 chunks, ~2372 entites extraites.
