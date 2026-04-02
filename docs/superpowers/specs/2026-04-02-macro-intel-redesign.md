# Spec — Macro Intelligence Dashboard Redesign
**Date :** 2026-04-02
**Projet :** sentiment_platform
**Priorité :** B (Corpus) > A (Tweets) > C (Digest) > D (Visual)

---

## 1. Contexte & Objectif

Transformer le prototype existant (4 phases déjà buildées) en un outil d'analyse macro opérationnel pour 9 traders, exploitant deux sources réelles :

- **`data_tweet/financial_juice_tweets.csv`** — 607 tweets FinancialJuice (Mar 28–31 2026), source unique, thèmes dominants : Iran/Hormuz/Oil, ECB rates, Germany macro
- **`data_corpus/*.pdf`** — 14 documents macro réels (Goldman Sachs, BofA, Macquarie, Natixis, SEB, Alexander Campbell, Cavendish, DBS, etc.)

**Contrainte clé :** zéro LLM payant ou API externe. Tout NLP tourne en local.

---

## 2. Stack technique retenue

| Couche | Outil | Justification |
|--------|-------|---------------|
| Sentiment | `FinBERT` (ProsusAI, existant) | Spécialisé finance, local, gratuit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 80MB, offline, partagé tweets+corpus → comparaison cross-source |
| Topic modeling | `BERTopic` | Topics cohérents sans définir N, visualisations Plotly natives |
| Vectorstore corpus | `ChromaDB` (persist_directory) | In-process, persisté sur disque → ingest une fois, lecture instantanée |
| Consensus/divergence | Cosine similarity + marqueurs lexicaux | Interprétable, zéro GPU requis |
| Dashboard | `Streamlit` (existant) | Python natif, time-to-demo rapide |
| Prix marché | `yfinance` + Alpha Vantage (existant) | Gratuit |
| DB metadata | `SQLite` via SQLAlchemy (existant) | Conservé pour signaux et metadata |

---

## 3. Architecture des modules

### 3.1 Nouveaux modules

```
module1_twitter/twitter/csv_backend.py
    → Lit data_tweet/financial_juice_tweets.csv
    → Normalise au format standard {id, created_at, author, followers_count,
      text, retweet_count, like_count}
    → followers_count = 0 (source unique FinancialJuice)

module2_nlp/analysis/
    ├── __init__.py
    ├── topic_model.py       → BERTopic wrapper (fit + transform + visualize)
    ├── consensus.py         → cosine similarity + marqueurs lexicaux
    ├── cross_source.py      → comparaison embeddings tweets vs corpus
    └── corpus_store.py      → ChromaDB ingest + semantic search

scripts/
    └── ingest_corpus.py     → CLI : parse tous les PDFs data_corpus/,
                               chunk, embed, store ChromaDB + SQLite
```

### 3.2 Modules existants conservés (sans modification majeure)

```
shared/nlp/vader_sentiment.py         ✅ conservé
shared/nlp/finbert_sentiment.py       ✅ conservé (singleton)
shared/backtest/price_client.py       ✅ conservé
shared/backtest/engine.py             ✅ conservé
shared/db/database.py                 ✅ conservé + 1 nouvelle table corpus_chunks
module2_nlp/ingestion/pdf_parser.py   ✅ conservé (utilisé par ingest_corpus.py)
module2_nlp/nlp/chunker.py            ✅ conservé
module2_nlp/nlp/ner.py                ✅ conservé
module2_nlp/nlp/pipeline.py           ✅ conservé (utilisé par Tab 3)
```

### 3.3 Modules supprimés / remplacés

```
module1_twitter/twitter/mock_backend.py    → remplacé par csv_backend
module2_nlp/ingestion/mock_documents.py   → remplacé par vrais PDFs
```

---

## 4. Pipeline de données

### 4.1 Tweets (chargement au démarrage du dashboard)

```
financial_juice_tweets.csv
    → csv_backend.py : parse + normalise (600ms)
    → preprocessor.clean_tweet() : nettoyage
    → FinBERT/VADER : sentiment score par tweet
    → BERTopic : topic assignment (fit une fois, @st.cache_resource)
    → consensus.classify_tweet() : label consensus/divergence/signal_faible
    → DataFrame enrichi en mémoire (résultat mis en cache session)
```

### 4.2 Corpus (ingest une fois, lecture instantanée ensuite)

```
data_corpus/*.pdf
    → pdf_parser.py : extraction texte (PyMuPDF)
    → chunker.py : chunks 200 mots, overlap 50
    → sentence-transformers : embeddings par chunk
    → ChromaDB (persist_directory="shared/db/chroma/") : stockage
    → ner.py + FinBERT : sentiment par entité par document
    → SQLite : metadata documents + signaux entités

Détection nouveaux fichiers : liste des PDFs déjà ingérés stockée
dans SQLite (table corpus_ingested). Script idempotent :
    - Si PDF déjà ingéré (même nom de fichier) → skip
    - Si nouveau PDF → process uniquement lui → append ChromaDB
    → Bouton "🔄 Re-ingest" dans Tab 3 sidebar déclenche le script
```

### 4.3 Cross-source analysis (calculé à la demande, mis en cache)

```
Pour chaque topic BERTopic :
    - Récupérer embeddings des tweets du topic
    - Récupérer embeddings des chunks corpus sur même thème (ChromaDB query)
    - Cosine similarity cross-source → score alignement 0→1
    - < 0.4 = divergent, 0.4–0.7 = mixte, > 0.7 = aligné
```

---

## 5. Dashboard — 4 tabs

### Tab 1 — 🧭 Macro Digest

**Objectif :** lecture complète en 60 secondes.

**KPI Row (5 colonnes) :**
- Tweets analysés | Docs ingérés | Topics détectés | Consensus | Divergences

**Row 2 (2 colonnes) :**
- **Gauche — Heatmap entités × sources** : lignes = entités trackées (Oil, Gold, Fed, ECB, Iran, OPEC, China, Rates, Inflation…), colonnes = 14 docs + "Tweets CSV". Couleur = sentiment score (rouge -1 → vert +1). Cases vides si entité non mentionnée.
- **Droite — Digest narratif automatique** :
  - 📊 Consensus Views (top 3 : entités où ≥ 3 sources s'accordent, variance < 0.15)
  - ⚡ Divergences (top 3 : entités où delta inter-sources > 0.5)
  - 🔍 Signaux faibles (top 2 : thèmes présents dans corpus mais absents tweets, ou inversement)

**Row 3 — Cross-source alignment bar chart :**
- Par topic : score alignement corpus ↔ tweets (0→1)
- Rouge = divergent, vert = aligné

---

### Tab 2 — 📡 Tweet Intelligence

**Sidebar :**
- `st.multiselect` Topic (labels BERTopic)
- `st.multiselect` Sentiment (Positive / Negative / Neutral)
- `st.multiselect` Type (Consensus / Divergence / Signal faible)
- `st.slider` Date range (4 jours disponibles)

**Contenu :**
1. **Timeline sentiment** (line chart) : évolution positive/negative/neutral sur 4 jours. Marqueurs rouges sur pics d'intensité (|signal| > 0.7).
2. **Row 2 (2 colonnes) :**
   - Feed scrollable paginé (50/page) : `🔴/🟢/⚪ badge_type | texte | score | topic`
   - Topic Map BERTopic 2D scatter (clic sur cluster → filtre feed)
3. **Top keywords par topic** (bar chart horizontal)

---

### Tab 3 — 📚 Corpus Analysis

**Sidebar :**
- `st.multiselect` Documents (tous par défaut)
- `st.selectbox` Doc type filter
- `st.selectbox` Entité focus
- `st.radio` Modèle VADER / FinBERT
- `st.button("🔄 Re-ingest corpus")` → relance `ingest_corpus.py`

**Contenu :**
1. **Heatmap Documents × Entités** : lignes = 14 docs (nom court), colonnes = entités trackées. Couleur = score sentiment. Cellule vide si entité non mentionnée dans ce doc.
2. **Row 2 (2 colonnes) :**
   - Thèmes prédéfinis (Oil/Rates/Geopolitics/Equities/China) : bar chart horizontal, score moyen pondéré par mentions
   - Signaux émergents : mots fréquents hors taxonomy (TF-IDF résiduel) → bar chart top 10
3. **Tableau divergences inter-documents** : colonnes Entity | Doc A | Score A | Doc B | Score B | Δ. Trié par Δ décroissant. Seuil Δ > 0.4.
4. **Recherche sémantique** : `st.text_input` → ChromaDB similarity search → top 5 passages (source, extrait, score similarité cosinus)

---

### Tab 4 — 📊 Backtest

**Contrôles (sidebar) :**
- `st.radio` Source : Tweets CSV / Corpus Documents
- `st.selectbox` Asset prix (WTI/Brent/Gold/SPX/NASDAQ/EUR-USD)
- `st.selectbox` Entité/Topic (si Source = Corpus)
- `st.select_slider` Horizon (15min → 1w)
- `st.date_input` start / end
- `st.info` auto source (Alpha Vantage <1h, yfinance ≥1h)
- `st.button("▶ Run Backtest")`

**Résultats :**
1. **5 KPIs** : Directional Accuracy | Pearson r | Spearman r | N signals | p-value
2. **Forward Returns by Bucket** (bar rouge/vert)
3. **Row 2 (2 colonnes)** : Rolling Correlation (line, 20 obs) | Scatter signal vs return + OLS trendline

---

## 6. Livrables attendus

| Livrable | Quand |
|----------|-------|
| `notes_choix_techniques.md` | Avant le dev (résume les arbitrages clés) |
| `README.md` | Mis à jour en continu, instructions de lancement complètes |
| `DOCUMENTATION.md` | Mis à jour — supprime mock_backend/mock_docs, ajoute nouveaux modules |
| `CLAUDE.md` | Mis à jour avec la nouvelle architecture |
| Code + dashboard fonctionnel | Fin de session |

---

## 7. Nouvelles dépendances (à ajouter à requirements.txt)

```
sentence-transformers>=2.2
bertopic>=0.16
chromadb>=0.4
langchain-community>=0.0.20   # optionnel, pour évolution RAG
```

---

## 8. Contraintes & limites connues

- **FinBERT 512 tokens max** : les chunks corpus passent par le chunker existant (200 mots) → pas de problème
- **BERTopic nécessite ~20 docs min** : 607 tweets → OK. Sur corpus 14 docs, on utilise TF-IDF plutôt que BERTopic
- **ChromaDB persistence** : si `shared/db/chroma/` supprimé → relancer ingest. Documenté dans README.
- **Tweets FinancialJuice** : source unique, pas de follower count → signal extractor désactivé pour la pondération (tous les tweets ont le même poids)
- **Ollama Q&A** : non inclus dans ce sprint, architecture prête (ChromaDB retriever disponible)
