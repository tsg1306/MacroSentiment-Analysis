# Documentation — Macro Intelligence Platform

Reference technique des modules, fonctions et signatures API.

---

## 1. `dashboard/app.py`

Dashboard Streamlit avec 4 onglets. Fichier unique, ~800 lignes.

### Fonctions cached

| Fonction | Signature | Description |
|---|---|---|
| `get_vader()` | `() -> VaderSentiment` | Cache et retourne l'analyseur VADER |
| `get_finbert()` | `() -> FinBERTSentiment` | Cache et retourne l'analyseur FinBERT (singleton) |
| `get_analyzer(name)` | `(str) -> VaderSentiment | FinBERTSentiment` | Retourne l'analyseur selon "VADER" ou "FinBERT" |
| `load_enriched_tweets(model_name)` | `(str) -> pd.DataFrame` | Charge tweets CSV, enrichit avec sentiment + consensus + theme. Cache 1h |
| `load_tweet_topics(model_name)` | `(str) -> tuple` | Fit BERTopic sur tweets. Retourne (model, topics_list, labels_dict). Cache 1h |
| `load_corpus_analyses()` | `() -> list[dict]` | Charge toutes les analyses de documents depuis SQLite. Cache 1h |
| `load_corpus_entity_signals()` | `() -> pd.DataFrame` | Charge sentiments par entite depuis SQLite (join documents). Cache 1h |
| `build_entity_heatmap(tweets_df, corpus_df, max_entities)` | `(DataFrame, DataFrame, int=10) -> DataFrame` | Construit pivot table entite x source pour heatmap. Top N entites par mentions |

### 4 Onglets

- **Tab 1 — Macro Digest** : KPIs + heatmap entite x source + digest 3 colonnes (consensus, trades, weak signals) + cross-source alignment
- **Tab 2 — Tweet Intelligence** : filtres sidebar (theme, sentiment, type, dates) + timeline horaire + feed pagine + donut + topic chips + top entities
- **Tab 3 — Corpus Analysis** : heatmap docs x entites + summaries expandables + filtres stance/trade/domaine + divergences inter-docs + recherche semantique ChromaDB
- **Tab 4 — Backtest** : source tweets/corpus + asset + horizon + prix + sentiment overlay + KPIs + scatter OLS

---

## 2. `shared/nlp/`

### `vader_sentiment.py` — Classe `VaderSentiment`

| Methode | Description |
|---|---|
| `__init__()` | Initialise VADER + enrichit le lexique avec 12 termes financiers (bullish +3, crash -3.5, surge +2.5, ceasefire +1.5, sanctions -2, etc.) |
| `analyze(text) -> dict` | Retourne `{score, label, confidence, model:"vader"}`. Seuils : >0.05 = positive, <-0.05 = negative |

### `finbert_sentiment.py` — Classe `FinBERTSentiment` (singleton)

| Methode | Description |
|---|---|
| `__new__(cls)` | Pattern singleton : charge `ProsusAI/finbert` une seule fois. Fallback VADER si torch/transformers absent |
| `analyze(text) -> dict` | Retourne `{score, label, confidence, model:"finbert"}`. Score = P(positive) - P(negative). Truncation 512 tokens |

**Interface commune** : les deux classes retournent exactement `{score: float[-1,1], label: str, confidence: float[0,1], model: str}`

---

## 3. `shared/backtest/`

### `price_client.py` — Classe `PriceClient`

| Methode | Signature | Description |
|---|---|---|
| `get_prices()` | `(asset, start, end, horizon_hours) -> pd.DataFrame` | Routage : <1h -> Alpha Vantage 1min, 1-24h -> yfinance 1h, >24h -> yfinance 1d. Index DatetimeIndex UTC, colonne `close` |
| `_resolve_ticker()` | `(asset) -> dict` | Cherche dans `ASSETS` puis `ENTITY_TICKERS`. Raise `KeyError` si inconnu |
| `_fetch_yfinance()` | `(ticker, start, end, interval) -> DataFrame` | Appel yfinance, gere MultiIndex et timezone |
| `_fetch_alpha_vantage()` | `(symbol, start, end) -> DataFrame` | Appel API Alpha Vantage intraday 1min |

### `engine.py` — Classe `BacktestEngine`

| Methode | Signature | Description |
|---|---|---|
| `run()` | `(signals, asset, horizon_hours=24) -> dict` | Backtest complet. Retourne `{forward_returns_by_bucket, rolling_correlation, directional_accuracy, accuracy_by_quintile, confusion_matrix, pearson_r, pearson_pval, spearman_r, n_signals, raw_df}`. Retourne `{"error": ...}` si <5 signaux |
| `_closest_price()` | `(prices, target_time) -> float | None` | Prix le plus proche d'un timestamp |
| `_bucket_returns()` | `(df) -> dict` | Forward returns moyens par bucket (Very Negative -> Very Positive) |
| `_rolling_correlation()` | `(df, window=20) -> list` | Correlation glissante signal/return |
| `_directional_accuracy()` | `(signals, returns) -> float` | % de fois ou sign(signal) == sign(return) |
| `_confusion_matrix()` | `(signals, returns) -> dict` | `{TP, TN, FP, FN}` |

---

## 4. `shared/db/database.py`

SQLAlchemy ORM + CRUD. 7 tables.

### Tables

| Table | PK | Description |
|---|---|---|
| `tweets` | `id` (TEXT) | Tweets individuels avec sentiment |
| `tweet_signals` | `id` (AUTO) | Signaux agreges par fenetre temporelle |
| `documents` | `id` (TEXT, SHA256[:16]) | Documents ingeres (metadata) |
| `entity_sentiments` | `id` (AUTO) | Sentiment par entite et chunk |
| `document_signals` | `id` (AUTO) | Signal agrege par entite et document |
| `corpus_ingested` | `filename` (TEXT) | Tracking des PDFs deja ingeres |
| `document_analysis` | `doc_id` (TEXT, FK) | Classification domaine/stance/trade + resume |

### Fonctions CRUD

| Fonction | Description |
|---|---|
| `init_db()` | Cree toutes les tables si inexistantes |
| `get_session()` | Retourne une session SQLAlchemy |
| `save_tweet(dict)` | INSERT OR IGNORE (idempotent) |
| `save_tweet_signal(dict)` | Insert signal agrege Twitter |
| `save_document(dict)` | INSERT OR IGNORE (idempotent sur SHA256) |
| `save_entity_sentiment(dict)` | Insert sentiment par entite |
| `save_doc_signal(dict)` | Insert signal document |
| `save_document_analysis(dict)` | Upsert analyse document (domaine/stance/trade/summary) |
| `get_tweets(asset?, since_minutes?)` | Filtre par asset et/ou fenetre |
| `get_tweet_signals(asset?)` | Signaux Twitter |
| `get_doc_signals(entity?, doc_type_filter?)` | Signaux documents filtres |
| `get_all_entities()` | Liste unique des entites |
| `get_ingested_files()` | Set des filenames ingeres |
| `mark_file_ingested(filename)` | Marque un PDF comme ingere |
| `get_document_analysis(doc_id)` | Retourne analyse d'un document |
| `get_all_document_analyses(domain_filter?, stance_filter?, trade_filter?)` | Toutes les analyses avec filtres optionnels. Outerjoin avec documents |

---

## 5. `module1_twitter/twitter/`

### `client.py` — Classe `TwitterClient` (factory)

| Methode | Description |
|---|---|
| `__init__(backend="csv")` | Instancie : `CsvBackend`, `MockBackend`, `SnscrapeBackend`, ou `ApiBackend` |
| `search(keywords, limit=100)` | Recherche par mot-cle |
| `get_user_tweets(username, limit=10)` | Recuperation par compte |
| `get_all()` | Tous les tweets (CsvBackend seulement) |

### `csv_backend.py` — Classe `CsvBackend`

Charge `data_tweet/financial_juice_tweets.csv` (492 tweets exploitables).

Format retourne : `{id, created_at, author, followers_count, text, retweet_count, like_count}`

---

## 6. `module1_twitter/nlp/preprocessor.py`

| Fonction | Description |
|---|---|
| `clean_tweet(text) -> str` | Pipeline : lowercase -> strip URLs -> strip @mentions -> hashtags sans # -> synonymes (crude oil->oil, petroleum->oil, xau/usd->gold, federal reserve->Fed) -> strip caracteres speciaux -> normalise espaces |

---

## 7. `module1_twitter/signal/extractor.py`

| Fonction | Description |
|---|---|
| `compute_weighted_signal(tweets, window_minutes, asset) -> dict` | Filtre par asset_tag + fenetre. Poids = `log(followers+1) * (1+log(retweets+1)) * authority`. Retourne `{signal, tweet_count, alert}` |
| `compute_all_windows(tweets, asset) -> dict` | Signal pour chaque fenetre (60/240/1440 min) |

---

## 8. `module2_nlp/ingestion/`

### Parsers

| Classe | Fichier | Input | Specificite |
|---|---|---|---|
| `BaseParser` | `base_parser.py` | Abstract | Format retour : `{title, source, doc_type, published_at, text}` |
| `PdfParser` | `pdf_parser.py` | Chemin ou bytes | PyMuPDF (fitz) |
| `HtmlParser` | `html_parser.py` | URL ou HTML | BeautifulSoup, extrait published_time |
| `TxtParser` | `txt_parser.py` | str ou bytes | Texte brut UTF-8 |

---

## 9. `module2_nlp/nlp/`

### `chunker.py`

| Fonction | Description |
|---|---|
| `chunk_text(text, chunk_size=200, overlap=50) -> list[str]` | Decoupe en chunks respectant les limites de phrases |

### `ner.py`

| Fonction | Description |
|---|---|
| `extract_entities_with_context(text) -> list[dict]` | spaCy NER (ORG, PERSON, GPE, PRODUCT, MONEY) + TRACKED_ENTITIES custom. Deduplique par (entity, type). Retourne `{entity, entity_type, context_text}` |

### `pipeline.py`

| Fonction | Description |
|---|---|
| `process_document(text, file_type, doc_type, model, title, source, published_at) -> dict` | Pipeline complet : SHA256 doc_id -> chunk -> NER -> sentiment par chunk -> save DB. Idempotent. Retourne `{document, signals}` |

---

## 10. `module2_nlp/signal/aggregator.py`

| Fonction | Description |
|---|---|
| `aggregate_signals(signals, entity?, doc_type_filter?) -> dict` | Moyenne ponderee par mention_count. Retourne `{signal, n_docs, total_mentions}` |

---

## 11. `module2_nlp/analysis/`

### `corpus_store.py` — ChromaDB wrapper

Persist directory : `shared/db/chroma/`. Embeddings : all-MiniLM-L6-v2 (384 dims).

| Fonction | Description |
|---|---|
| `ingest_chunks(chunks, metadatas, doc_id) -> int` | Embed + upsert dans ChromaDB. IDs : `{doc_id}_chunk_{i}`. Idempotent |
| `semantic_search(query, n_results=5, where=None) -> list[dict]` | Recherche semantique. Retourne `{text, source, doc_type, similarity, doc_id}` |
| `get_corpus_theme_embedding(keywords) -> np.ndarray | None` | Embedding moyen des chunks matchant des keywords |
| `get_collection_count() -> int` | Nombre de chunks en collection |
| `delete_collection()` | Supprime la collection (pour --force re-ingest) |

### `topic_model.py` — BERTopic wrapper

| Fonction | Description |
|---|---|
| `fit_tweet_topics(texts, n_topics=8) -> tuple` | BERTopic + KMeans. Retourne `(model, topics_list, labels_dict)` |
| `get_topic_barchart(model, top_n=8) -> Figure | None` | Plotly barchart keywords par topic |
| `get_topic_map(model) -> Figure | None` | Plotly scatter 2D des topics |

### `consensus.py`

| Fonction | Description |
|---|---|
| `classify_tweet(text) -> dict` | Classification par marqueurs lexicaux. Retourne `{label, consensus_score, divergence_score, signal_score}`. Labels : consensus / divergence / signal_faible / neutral |
| `detect_divergences(signals, threshold=0.4) -> list[dict]` | Divergences inter-sources (delta > threshold). Retourne `{entity, doc_a, score_a, doc_b, score_b, delta}` |
| `detect_consensus(signals, min_sources=2, max_std=0.20, min_abs_mean=0.25) -> list[dict]` | Consensus inter-sources. Retourne `{entity, mean_score, std, n_sources, direction}` |

### `cross_source.py`

| Fonction | Description |
|---|---|
| `compute_theme_alignment(keywords, tweet_texts, corpus_emb) -> float | None` | Cosine similarity tweets vs corpus pour un theme [0-1] |
| `get_all_theme_alignments(tweet_texts) -> dict` | Alignement pour les 7 themes |

### `document_classifier.py`

Classification sur 3 axes : domaine (7 themes), stance, trade signals.

| Fonction | Description |
|---|---|
| `classify_document(text, filename="") -> dict` | Retourne `{domains, primary_domain, stance, trade_signal, explicit_trades, implicit_trades}` |
| `_score_themes(text) -> dict` | Score TF-IDF par theme. Seuil 2.0 |
| `_detect_stance(text) -> str` | "institutional" / "investor" / "research_note" par marqueurs lexicaux |
| `_classify_sentence(sentence) -> str` | "explicit" / "implicit" / "none" par marqueurs trade |

**THEMES** : 6 themes actifs + "Sector / Other" fallback
**EXPLICIT_TRADE_MARKERS** : "we buy", "overweight", "target price", "outperform"...
**IMPLICIT_TRADE_MARKERS** : "attractive", "compelling", "upside", "we prefer"...

### `summarizer.py`

Resume extractif TF-IDF avec priorite trade.

| Fonction | Description |
|---|---|
| `generate_summary(text, max_bullets=10) -> list[dict]` | Retourne `[{text, type}]`. Types : "explicit" / "implicit" / "info". Priorite : explicit d'abord, puis implicit, puis info (TF-IDF ranked) |

Parametres internes : MIN_SENTENCE_LEN=25, MAX_SENTENCE_LEN=350.

---

## 12. `scripts/ingest_corpus.py`

CLI d'ingestion des PDFs. Usage : `python scripts/ingest_corpus.py [--force] [--model vader|finbert]`

| Fonction | Description |
|---|---|
| `guess_doc_type(filename) -> str` | Heuristique filename -> doc_type (research_note, hedge_fund_letter, fomc_minutes...) |
| `clear_corpus_ingested_table()` | Vide corpus_ingested + document_analysis (--force) |
| `ingest(force, model)` | Pour chaque PDF : parse -> chunk -> ChromaDB -> NLP pipeline (NER+sentiment) -> classify_document -> generate_summary -> save_document_analysis -> mark_file_ingested |

Pipeline par document : ~10 etapes, ~20s par doc (varie selon taille).
