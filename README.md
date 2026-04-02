# Macro Intelligence — Prototype d'analyse macro multi-sources

Prototype d'analyse macro combinant un **flux de tweets** et un **corpus de documents financiers** pour produire une lecture synthetique et actionnable de l'actualite macro.

## Donnees exploitees

| Source | Volume | Periode |
|--------|--------|---------|
| Tweets FinancialJuice (`data_tweet/`) | ~492 tweets exploitables (607 lignes brutes) | 28-31 Mars 2026 |
| Corpus macro PDF (`data_corpus/`) | 14 documents (Goldman Sachs, BofA, Macquarie, Natixis, SEB, Canaccord, Cavendish, DBS, etc.) | Mars 2026 |

## Lancement rapide

```bash
# 1. Clone
git clone https://github.com/tsg1306/MacroSentiment-Analysis.git
cd sentiment_platform

# 2. Dependances
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configuration (optionnel)
cp .env.example .env
# Editer .env si besoin (Alpha Vantage key pour backtest intraday <1h)

# 4. Initialisation DB + ingestion du corpus
python -c "from shared.db.database import init_db; init_db()"
python scripts/ingest_corpus.py
# -> Parse 14 PDFs, ~630 chunks vectorises (ChromaDB), ~2700 entites (SQLite)
# -> Classifie chaque doc : domaine, stance, trade signals, 10-bullet summary
# -> Duree : ~3 min (NER spaCy + sentence-transformers embeddings)

# 5. Lancer le dashboard
python -m streamlit run dashboard/app.py
```

## Ce que fait le prototype

### Pipeline d'analyse automatise

Le pipeline `ingest_corpus.py` enchaine pour chaque document :

1. **Parsing PDF** (PyMuPDF) → texte brut
2. **Chunking** (200 mots, overlap 50) → respecte les limites de phrases
3. **Vectorisation** (sentence-transformers all-MiniLM-L6-v2) → ChromaDB pour recherche semantique
4. **NER** (spaCy en_core_web_sm + entites trackees custom) → extraction des entites financieres
5. **Sentiment par entite** (FinBERT ProsusAI ou VADER) → score [-1, +1] par chunk et entite
6. **Classification thematique** (TF-IDF, 7 domaines) → Macro/Rates, Oil/Energy, Geopolitics, Equities/Risk, China/EM, Europe/FX, Sector/Other
7. **Detection de stance** (marqueurs lexicaux) → institutional / investor / research_note
8. **Extraction de signaux trade** (marqueurs lexicaux par phrase) → explicit / implicit / none
9. **Resume extractif** (TF-IDF sentence ranking) → 10 bullets, tries par priorite (explicit > implicit > info)
10. **Persistance** → SQLite (metadata, entites, signaux) + ChromaDB (vecteurs)

Pour les tweets, le pipeline enrichit chaque tweet avec : sentiment, topic (BERTopic), classification consensus/divergence/signal faible.

### Dashboard interactif — 4 onglets

#### Tab 1 — Macro Digest
Vue synthetique combinant les deux sources :
- **5 KPIs** en un coup d'oeil : tweets analyses, docs ingeres, topics, consensus, divergences
- **Heatmap entite x source** : sentiment par entite (oil, gold, fed, iran...) croise avec les 14 docs + tweets — top 10 entites les plus mentionnees
- **Digest en 3 colonnes** :
  - Consensus Views (entites ou les sources convergent, ex: bearish oil)
  - Trade Signals (phrases explicites/implicites extraites des docs : "overweight gold", "we prefer...")
  - Weak Signals (entites presentes dans le corpus mais peu couvertes par les tweets)
- **Cross-source alignment** : score d'alignement [0-1] entre tweets et corpus par theme, identifie les zones ou les deux sources divergent

#### Tab 2 — Tweet Intelligence
Exploration interactive du flux :
- **Filtres** : theme, sentiment, type (consensus/divergence/signal faible), texte libre, dates
- **Timeline horaire** : evolution du volume par sentiment (positive/negative/neutral)
- **Feed pagine** : 50 tweets/page avec bordure coloree par sentiment + scores
- **Panel lateral** : donut sentiment, topic chips colores, top 5 entites mentionnees

#### Tab 3 — Corpus Analysis
Analyse structuree des 14 documents :
- **Heatmap docs x entites** : score sentiment par document, badges trade [X] explicit / [I] implicit
- **Summaries expandables** : pour chaque document, 10 bullets extractifs avec code couleur (rouge=explicit trade, jaune=implicit, gris=info)
- **Filtres** : stance (institutional/investor/research), trade signal, domaine thematique
- **Divergences inter-docs** : tableau des entites ou les documents divergent (delta > 0.5)
- **Recherche semantique** : query texte libre → top 5 passages les plus proches (cosine similarity ChromaDB)

#### Tab 4 — Backtest
Validation experimentale du pouvoir predictif du sentiment :
- **Sources** : Tweets CSV ou Corpus Documents
- **Assets** : WTI, Brent, Gold, SPX, NASDAQ, EUR/USD
- **Horizons** : 15min a 1 semaine
- **Resultats** : Directional Accuracy, Pearson/Spearman r, p-value, scatter + OLS trendline, prix + barres de sentiment superposees
- Note : resultats indicatifs sur 4 jours de donnees, non valides statistiquement

## Architecture

```
sentiment_platform/
├── config.py                              # Config centralisee (assets, seuils, modeles)
├── shared/
│   ├── nlp/                               # VADER + FinBERT (singleton)
│   ├── backtest/                          # PriceClient (yfinance/AV) + BacktestEngine
│   └── db/                                # SQLite (7 tables) + ChromaDB (chroma/)
├── module1_twitter/
│   ├── twitter/                           # CSV backend + client factory
│   ├── nlp/                               # Preprocessor tweets
│   └── signal/                            # Signal extractor
├── module2_nlp/
│   ├── ingestion/                         # Parsers PDF/HTML/TXT
│   ├── nlp/                               # Chunker + NER + Pipeline
│   ├── signal/                            # Aggregator multi-docs
│   └── analysis/
│       ├── corpus_store.py                # ChromaDB wrapper (embed, search)
│       ├── topic_model.py                 # BERTopic + KMeans
│       ├── consensus.py                   # Consensus / divergence detection
│       ├── cross_source.py                # Alignement tweets <-> corpus
│       ├── document_classifier.py         # 7 themes + stance + trade extraction
│       └── summarizer.py                  # Resume extractif TF-IDF
├── dashboard/
│   └── app.py                             # Streamlit 4 tabs
├── scripts/
│   └── ingest_corpus.py                   # CLI: ingest PDFs → ChromaDB + SQLite
├── data_tweet/                            # financial_juice_tweets.csv
└── data_corpus/                           # 14 PDFs macro
```

## Stack technique

| Couche | Outil | Justification |
|--------|-------|---------------|
| Sentiment | FinBERT (ProsusAI) + VADER | FinBERT fine-tune sur 4.9M phrases financieres, VADER en fallback rapide |
| Embeddings | all-MiniLM-L6-v2 | 384 dims, 80MB, espace vectoriel partage tweets/corpus |
| Topic modeling | BERTopic + KMeans | Topics semantiquement coherents sur textes courts |
| Vector store | ChromaDB (local) | Persistance native, recherche semantique, filtres metadata |
| NER | spaCy en_core_web_sm + custom | Entites financieres trackees : commodities, macro, geopolitics |
| Classification docs | TF-IDF keyword scoring | 7 themes, stance par marqueurs lexicaux, trade par phrase |
| Prix marche | yfinance + Alpha Vantage | yfinance >=1h, Alpha Vantage <1h (intraday) |
| DB | SQLite (SQLAlchemy) | 7 tables, CRUD idempotent |
| Dashboard | Streamlit + Plotly | Prototypage rapide, interactivite, rendu pro |

## Variables d'environnement (.env)

```env
TWITTER_BACKEND=csv              # csv | mock | snscrape | api
ALPHA_VANTAGE_KEY=               # requis si horizon < 1h en backtest
SENTIMENT_MODEL=vader            # vader | finbert (defaut sidebar)
DB_PATH=shared/db/sentiment.db
```

## Re-ingestion du corpus

```bash
# Ajouter un PDF dans data_corpus/, puis :
python scripts/ingest_corpus.py          # incremental (skip fichiers deja traites)
python scripts/ingest_corpus.py --force  # tout re-ingerer
python scripts/ingest_corpus.py --model finbert  # avec FinBERT

# Ou depuis le dashboard : Tab 3 → bouton "Re-ingest corpus"
```

## Documentation complementaire

- `notes_choix_techniques.md` — Principaux choix techniques et arbitrages
- `DOCUMENTATION.md` — Signatures API detaillees de tous les modules
