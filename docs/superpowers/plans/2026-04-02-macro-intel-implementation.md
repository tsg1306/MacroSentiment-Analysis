# Macro Intelligence Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer le prototype existant en un outil d'analyse macro opérationnel exploitant 607 tweets FinancialJuice réels + 14 PDFs, avec un dashboard restructuré en 4 tabs (Macro Digest, Tweet Intelligence, Corpus Analysis, Backtest).

**Architecture:** Pipeline en deux couches — (1) ingest script one-shot pour les PDFs vers SQLite + ChromaDB, idempotent via table `corpus_ingested`; (2) chargement tweets CSV au démarrage dashboard (~1s). Dashboard lit uniquement depuis la DB/cache — jamais de traitement lourd à la demande.

**Tech Stack:** Python 3.11, Streamlit, FinBERT/VADER (existants), sentence-transformers `all-MiniLM-L6-v2`, BERTopic, ChromaDB, SQLite/SQLAlchemy (existant), Plotly, yfinance.

**Priorité d'exécution :** Tasks 0→8 sont le chemin critique. Tasks 9→12 sont le dashboard. Task 13 = livrables finaux.

---

## Fichiers créés / modifiés

| Fichier | Action |
|---------|--------|
| `requirements.txt` | Modifié — ajout sentence-transformers, bertopic, chromadb |
| `shared/db/database.py` | Modifié — ajout table `corpus_ingested` + 2 fonctions |
| `module1_twitter/twitter/csv_backend.py` | Créé |
| `module1_twitter/twitter/client.py` | Modifié — ajout backend "csv" |
| `module2_nlp/analysis/__init__.py` | Créé |
| `module2_nlp/analysis/corpus_store.py` | Créé |
| `module2_nlp/analysis/consensus.py` | Créé |
| `module2_nlp/analysis/cross_source.py` | Créé |
| `module2_nlp/analysis/topic_model.py` | Créé |
| `scripts/ingest_corpus.py` | Créé |
| `dashboard/app.py` | Réécrit complet |
| `notes_choix_techniques.md` | Créé |
| `README.md` | Mis à jour |
| `DOCUMENTATION.md` | Mis à jour |
| `CLAUDE.md` | Mis à jour |

---

## Task 0 — Livrables texte : notes techniques + CLAUDE.md + DOCUMENTATION.md

**Files:**
- Create: `notes_choix_techniques.md`
- Modify: `CLAUDE.md`
- Modify: `DOCUMENTATION.md`
- Modify: `README.md`

- [ ] **Step 1 : Créer `notes_choix_techniques.md`**

```markdown
# Notes — Choix techniques & arbitrages

## Contexte
Prototype d'analyse macro combinant flux tweets FinancialJuice (607 tweets, 4 jours)
et corpus de 14 documents financiers (Goldman Sachs, BofA, Macquarie, Natixis, SEB…).
Contrainte : zéro LLM payant / API externe. Tout tourne en local.

## Arbitrages principaux

### 1. BERTopic pour le topic modeling des tweets
**Retenu :** BERTopic (HDBSCAN + c-TF-IDF)
**Écarté :** LDA, NMF classique
**Raison :** BERTopic ne nécessite pas de définir N topics à l'avance, produit des
topics sémantiquement cohérents sur des textes courts, et inclut des visualisations
Plotly natives (carte 2D, barchart). Sur 607 tweets finance, les topics émergent
naturellement (Oil/Iran, ECB/Rates, Geopolitics, China/PBOC…).

### 2. sentence-transformers `all-MiniLM-L6-v2` pour les embeddings
**Retenu :** all-MiniLM-L6-v2 (80MB, local)
**Écarté :** text-embedding-ada-002 (OpenAI, payant), TF-IDF pur
**Raison :** Espace vectoriel partagé tweets ↔ corpus → comparaison cross-source
directe par cosine similarity. Modèle léger, inference en <2s sur 607 tweets CPU-only.

### 3. ChromaDB avec persist_directory pour le corpus
**Retenu :** ChromaDB local persisté (shared/db/chroma/)
**Écarté :** Faiss (pas de persistance native), Pinecone/Weaviate (cloud)
**Raison :** Ingest one-shot, lecture instantanée au redémarrage. Recherche sémantique
sur le corpus sans LLM pour la Q&A. Supporte les métadonnées (source, doc_type).

### 4. Pipeline d'ingest séparé (script CLI) vs smart-cache automatique
**Retenu :** Script `scripts/ingest_corpus.py` + table `corpus_ingested` SQLite
**Écarté :** Hash-check automatique au démarrage dashboard (trop complexe, risque de bug)
**Raison :** Fiabilité > élégance sur un prototype 6h. Le script est idempotent
(skip les fichiers déjà ingérés), appelable via bouton dans le dashboard.
Ajouter un PDF = relancer le script, simple et traçable.

### 5. FinBERT comme modèle principal, VADER en fallback
**Retenu :** FinBERT (ProsusAI/finbert) pour sentiment
**Raison :** Fine-tuné sur 4,9M phrases financières. Beaucoup plus précis que VADER
sur le vocabulaire macro (hawkish, dovish, tightening…). Le singleton est déjà
implémenté dans le projet.

### 6. Consensus/divergence par règles NLP (sans LLM)
**Approche :** Trois couches complémentaires :
- Marqueurs lexicaux (dictionnaire consensus/divergence/signal faible)
- Variance inter-sources sur même entité (std > 0.3 → divergence)
- Cosine similarity cross-source (tweets vs corpus par thème)
**Raison :** Interprétable, explicable au jury, zéro coût, reproductible.

### 7. Dashboard Streamlit, pas FastAPI + React
**Raison :** Time-to-demo 10x plus rapide. Audience = traders, pas ingénieurs.
Pas besoin d'API REST pour un usage en équipe interne de 9 personnes.

## Limites connues et extensions naturelles
- **Ollama (LLM local)** : non inclus dans ce sprint mais l'architecture ChromaDB
  est prête pour ajouter un RetrievalQA LangChain + llama3.2 en <1h
- **Tweets statiques** : le CSV est un snapshot. Extension : APScheduler + polling
  FinancialJuice toutes les 5min
- **BERTopic** : nécessite ~20+ textes pour des topics stables. 607 tweets = OK.
  Sur 14 docs corpus, on utilise TF-IDF résiduel à la place.
- **FinBERT 512 tokens** : chunker existant (200 mots) gère ce cas automatiquement.
```

- [ ] **Step 2 : Mettre à jour `CLAUDE.md`** — Ajouter section nouvelle architecture après la section "Architecture Monorepo" existante, et mettre à jour les phases de développement pour refléter le redesign. Ajouter :

```markdown
---

## Redesign 2026-04-02 — Macro Intelligence Dashboard

### Nouveaux modules

```
module2_nlp/analysis/
├── __init__.py
├── corpus_store.py      # ChromaDB ingest + semantic search (persist: shared/db/chroma/)
├── consensus.py         # Marqueurs lexicaux + variance inter-sources + detect_divergences()
├── cross_source.py      # Alignement tweets ↔ corpus par thème (cosine similarity)
└── topic_model.py       # BERTopic wrapper — fit sur tweets, visualize_topics()

module1_twitter/twitter/csv_backend.py  # Lit data_tweet/financial_juice_tweets.csv

scripts/
└── ingest_corpus.py     # CLI idempotent : parse PDFs → SQLite + ChromaDB
                         # Vérification via table corpus_ingested (SQLite)
                         # Usage : python scripts/ingest_corpus.py [--force]
```

### Nouvelle table SQLite
`corpus_ingested` : filename (PK), ingested_at — trace les PDFs déjà traités

### Dashboard restructuré (4 tabs)
- Tab 1 🧭 Macro Digest : heatmap entités×sources + digest narratif + cross-source alignment
- Tab 2 📡 Tweet Intelligence : feed interactif + timeline + BERTopic topic map
- Tab 3 📚 Corpus Analysis : heatmap docs×entités + thèmes + divergences + recherche sémantique
- Tab 4 📊 Backtest : backtest tweets ou corpus sur asset prix (fusionné ex-Tab2+Tab4)

### Stack ajoutée
- sentence-transformers >= 2.2 (all-MiniLM-L6-v2, 80MB)
- bertopic >= 0.16
- chromadb >= 0.4

### Modules supprimés / désactivés
- mock_backend.py : conservé mais backend "mock" non exposé dans le nouveau dashboard
- mock_documents.py : conservé mais non utilisé (remplacé par PDFs réels)

### Livrables attendus
- notes_choix_techniques.md : arbitrages techniques (rédigé avant le dev)
- README.md : instructions de lancement complètes, mis à jour en continu
- DOCUMENTATION.md : architecture mise à jour
```

- [ ] **Step 3 : Mettre à jour `DOCUMENTATION.md`** — Supprimer les sections mock_backend et mock_documents, ajouter les nouveaux modules. Structure à garder : Architecture, Modules, Data Flow, Setup.

- [ ] **Step 4 : Mettre à jour `README.md`** — Remplacer le template initial par :

```markdown
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
| Twitter — CSV backend (FinancialJuice) | ⬜ TODO |
| Corpus — Ingest PDFs (ChromaDB + SQLite) | ⬜ TODO |
| Analysis — BERTopic topic modeling | ⬜ TODO |
| Analysis — Consensus/Divergence detection | ⬜ TODO |
| Analysis — Cross-source alignment | ⬜ TODO |
| Dashboard — Tab Macro Digest | ⬜ TODO |
| Dashboard — Tab Tweet Intelligence | ⬜ TODO |
| Dashboard — Tab Corpus Analysis | ⬜ TODO |
| Dashboard — Tab Backtest | ⬜ TODO |

## Setup

\`\`\`bash
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
\`\`\`

## Re-ingest si nouveau PDF ajouté

\`\`\`bash
# Ajouter le PDF dans data_corpus/, puis :
python scripts/ingest_corpus.py
# Le script skip automatiquement les fichiers déjà traités.
# Ou depuis le dashboard : Tab 3 sidebar → bouton "🔄 Re-ingest corpus"
\`\`\`

## Variables d'environnement (.env)

\`\`\`env
TWITTER_BACKEND=csv              # csv | mock | snscrape | api
ALPHA_VANTAGE_KEY=               # requis si horizon < 1h en backtest
SENTIMENT_MODEL=vader            # vader | finbert
DB_PATH=shared/db/sentiment.db
\`\`\`

## Données

- \`data_tweet/financial_juice_tweets.csv\` — 607 tweets FinancialJuice (Mar 28–31 2026)
- \`data_corpus/*.pdf\` — 14 documents macro (Goldman Sachs, BofA, Macquarie, Natixis, SEB…)

## Architecture

Voir \`CLAUDE.md\` pour la spec complète et \`notes_choix_techniques.md\` pour les arbitrages.
```

- [ ] **Step 5 : Commit**

```bash
git add notes_choix_techniques.md CLAUDE.md DOCUMENTATION.md README.md docs/
git commit -m "docs: add technical notes, update CLAUDE.md/README/DOCUMENTATION for redesign"
```

---

## Task 1 — Dépendances

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1 : Mettre à jour `requirements.txt`**

Ajouter après la ligne `scikit-learn` :

```
sentence-transformers>=2.2.0
bertopic>=0.16.0
chromadb>=0.4.0
umap-learn>=0.5.0
hdbscan>=0.8.0
```

- [ ] **Step 2 : Installer les dépendances**

```bash
cd /c/Users/ASUS/Desktop/github-project/sentiment_platform
pip install sentence-transformers bertopic chromadb umap-learn hdbscan
```

Attendre la fin. La première installation télécharge ~300MB (modèle inclus).

- [ ] **Step 3 : Vérifier les imports**

```bash
python -c "
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import chromadb
print('✅ All new dependencies OK')
"
```

Résultat attendu : `✅ All new dependencies OK`

- [ ] **Step 4 : Commit**

```bash
git add requirements.txt
git commit -m "deps: add sentence-transformers, bertopic, chromadb"
```

---

## Task 2 — DB : table corpus_ingested

**Files:**
- Modify: `shared/db/database.py`

- [ ] **Step 1 : Ajouter le modèle `CorpusIngested` et les deux fonctions**

Dans `shared/db/database.py`, après la classe `DocumentSignal` (ligne ~97), ajouter :

```python
class CorpusIngested(Base):
    __tablename__ = "corpus_ingested"
    filename = Column(Text, primary_key=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)
```

Après la fonction `get_all_entities()`, ajouter :

```python
def get_ingested_files() -> set:
    """Returns set of filenames already processed by ingest_corpus.py."""
    session = get_session()
    try:
        rows = session.query(CorpusIngested.filename).all()
        return {r[0] for r in rows}
    finally:
        session.close()


def mark_file_ingested(filename: str):
    """Mark a PDF filename as ingested. Idempotent."""
    session = get_session()
    try:
        existing = session.get(CorpusIngested, filename)
        if existing is None:
            session.add(CorpusIngested(filename=filename))
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 2 : Tester**

```bash
python -c "
from shared.db.database import init_db, get_ingested_files, mark_file_ingested
init_db()
assert get_ingested_files() == set()
mark_file_ingested('test.pdf')
assert 'test.pdf' in get_ingested_files()
mark_file_ingested('test.pdf')  # idempotent
assert len(get_ingested_files()) == 1
print('✅ corpus_ingested table OK')
"
```

- [ ] **Step 3 : Commit**

```bash
git add shared/db/database.py
git commit -m "feat: add corpus_ingested table with get/mark functions"
```

---

## Task 3 — CSV Backend

**Files:**
- Create: `module1_twitter/twitter/csv_backend.py`
- Modify: `module1_twitter/twitter/client.py`

- [ ] **Step 1 : Créer `module1_twitter/twitter/csv_backend.py`**

```python
import os
import sys
import hashlib
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data_tweet', 'financial_juice_tweets.csv'
)

_DATE_FMT = "%a %b %d %H:%M:%S +0000 %Y"


def _parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s.strip(), _DATE_FMT)
    except Exception:
        return datetime.utcnow()


def _make_id(row) -> str:
    raw = f"{row['date']}{row['content']}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


class CsvBackend:
    def __init__(self):
        path = os.path.abspath(CSV_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tweet CSV not found: {path}")
        self._df = pd.read_csv(path)
        self._df.columns = [c.strip() for c in self._df.columns]
        # Normalise to standard format
        self._tweets = []
        for _, row in self._df.iterrows():
            text = str(row.get('content', '')).strip()
            if not text or text.startswith('http'):
                continue
            self._tweets.append({
                "id":              _make_id(row),
                "created_at":      _parse_date(str(row.get('date', ''))),
                "author":          str(row.get('author_name', 'FinancialJuice')),
                "followers_count": 0,
                "text":            text,
                "retweet_count":   0,
                "like_count":      0,
            })

    def search(self, keywords: list, limit: int = 100) -> list:
        if not keywords:
            return self._tweets[:limit]
        kw_lower = [k.lower() for k in keywords]
        results = [
            t for t in self._tweets
            if any(kw in t["text"].lower() for kw in kw_lower)
        ]
        return results[:limit]

    def get_user_tweets(self, username: str, limit: int = 10) -> list:
        results = [t for t in self._tweets if t["author"].lower() == username.lower()]
        return results[:limit]

    def get_all(self) -> list:
        """Returns all tweets (used by dashboard for full pipeline)."""
        return self._tweets
```

- [ ] **Step 2 : Ajouter "csv" dans `module1_twitter/twitter/client.py`**

Modifier le bloc `__init__` pour ajouter le cas `"csv"` :

```python
elif backend == "csv":
    from module1_twitter.twitter.csv_backend import CsvBackend
    self.backend = CsvBackend()
```

Ajouter aussi la méthode `get_all()` dans `TwitterClient` :

```python
def get_all(self) -> list:
    """Returns all tweets — only available for csv backend."""
    if hasattr(self.backend, 'get_all'):
        return self.backend.get_all()
    return []
```

- [ ] **Step 3 : Tester**

```bash
python -c "
from module1_twitter.twitter.client import TwitterClient

client = TwitterClient(backend='csv')
all_tweets = client.get_all()
print(f'Total tweets: {len(all_tweets)}')
assert len(all_tweets) > 500, f'Expected 500+, got {len(all_tweets)}'

required = ['id','created_at','author','followers_count','text','retweet_count','like_count']
assert all(k in all_tweets[0] for k in required), 'Missing keys'
assert isinstance(all_tweets[0]['created_at'], __import__('datetime').datetime)
assert isinstance(all_tweets[0]['followers_count'], int)

oil_tweets = client.search(['oil', 'crude', 'OPEC'], limit=50)
print(f'Oil tweets: {len(oil_tweets)}')
assert len(oil_tweets) > 0
print('✅ CSV backend OK')
"
```

- [ ] **Step 4 : Commit**

```bash
git add module1_twitter/twitter/csv_backend.py module1_twitter/twitter/client.py
git commit -m "feat: add CSV backend for FinancialJuice tweets"
```

---

## Task 4 — Corpus Store (ChromaDB)

**Files:**
- Create: `module2_nlp/analysis/__init__.py`
- Create: `module2_nlp/analysis/corpus_store.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/__init__.py`** (fichier vide)

- [ ] **Step 2 : Créer `module2_nlp/analysis/corpus_store.py`**

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'db', 'chroma')
)

_chroma_client = None
_collection = None
_embed_model = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            "corpus_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def add_document_chunks(doc_id: str, chunks: list, metadata: dict) -> int:
    """
    Embed chunks and store in ChromaDB.
    Idempotent: skips if doc_id already has entries.
    Returns number of chunks added (0 if skipped).
    """
    collection = _get_collection()

    # Check if already ingested via doc_id metadata
    existing = collection.get(where={"doc_id": doc_id}, limit=1)
    if existing and existing["ids"]:
        return 0

    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {**metadata, "doc_id": doc_id, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return len(chunks)


def semantic_search(query: str, n_results: int = 5) -> list:
    """
    Returns top-N relevant corpus passages for a query.
    Each result: {text, source, doc_type, title, similarity}
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    model = _get_embed_model()
    query_emb = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        output.append({
            "text":       results["documents"][0][i],
            "source":     meta.get("source", "unknown"),
            "doc_type":   meta.get("doc_type", ""),
            "title":      meta.get("title", ""),
            "filename":   meta.get("filename", ""),
            "similarity": float(1 - dist),
        })
    return output


def get_corpus_theme_embedding(theme_keywords: list, n_results: int = 20):
    """
    Returns mean embedding vector for corpus passages matching theme keywords.
    Returns None if corpus is empty.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return None

    model = _get_embed_model()
    query = " ".join(theme_keywords)
    query_emb = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(n_results, count),
        include=["embeddings"],
    )

    if not results.get("embeddings") or not results["embeddings"][0]:
        return None

    return np.array(results["embeddings"][0]).mean(axis=0)


def corpus_count() -> int:
    """Returns total number of chunks in ChromaDB."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
```

- [ ] **Step 3 : Tester (ChromaDB in-memory test)**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from module2_nlp.analysis.corpus_store import add_document_chunks, semantic_search, corpus_count

# Test add
n = add_document_chunks(
    doc_id='test_doc_001',
    chunks=['Oil prices rose sharply on OPEC cut news.', 'Gold surged as safe haven demand increased.'],
    metadata={'source': 'TestSource', 'doc_type': 'news', 'title': 'Test', 'filename': 'test.pdf'}
)
print(f'Added {n} chunks')
assert n == 2

# Test idempotency
n2 = add_document_chunks('test_doc_001', ['another chunk'], {'source': 'x', 'doc_type': 'news', 'title': '', 'filename': ''})
assert n2 == 0, 'Should skip already-ingested doc'

# Test search
results = semantic_search('OPEC oil production cut', n_results=2)
assert len(results) > 0
assert 'text' in results[0]
assert 'similarity' in results[0]
print(f'Search result: {results[0][\"text\"][:60]}... (sim={results[0][\"similarity\"]:.2f})')
print('✅ corpus_store OK')
"
```

- [ ] **Step 4 : Commit**

```bash
git add module2_nlp/analysis/__init__.py module2_nlp/analysis/corpus_store.py
git commit -m "feat: add ChromaDB corpus store with persistent embeddings"
```

---

## Task 5 — Script d'ingest corpus

**Files:**
- Create: `scripts/ingest_corpus.py`

- [ ] **Step 1 : Créer `scripts/ingest_corpus.py`**

```python
#!/usr/bin/env python
"""
Ingests all PDFs from data_corpus/ into SQLite (via pipeline) + ChromaDB.
Idempotent: skips already-processed files tracked in corpus_ingested table.

Usage:
    python scripts/ingest_corpus.py           # ingest new files only
    python scripts/ingest_corpus.py --force   # re-ingest all files
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from module2_nlp.ingestion.pdf_parser import PdfParser
from module2_nlp.nlp.pipeline import process_document
from module2_nlp.nlp.chunker import chunk_text
from module2_nlp.analysis.corpus_store import add_document_chunks
from shared.db.database import init_db, get_ingested_files, mark_file_ingested

CORPUS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data_corpus')
)

# Map filename keywords → doc_type
_DOC_TYPE_RULES = [
    (["fomc", "federal reserve", "fed minutes"], "fomc_minutes"),
    (["earnings", "q1 ", "q2 ", "q3 ", "q4 "], "earnings_call"),
    (["howell", "campbell", "bexelius", "cascade"], "hedge_fund_letter"),
    (["goldman", "bofа", "bof", "natixis", "macquarie",
      "canaccord", "cavendish", "dbs", "seb"], "research_note"),
]


def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    for keywords, doc_type in _DOC_TYPE_RULES:
        if any(kw in name for kw in keywords):
            return doc_type
    return "news"


def _infer_source(filename: str) -> str:
    """Extract source name from filename pattern 'Source-Title-Date.pdf'."""
    base = os.path.splitext(filename)[0]
    parts = base.split('-')
    return parts[0].replace('_', ' ') if parts else base


def _infer_title(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    parts = base.split('-')
    if len(parts) >= 2:
        return parts[1].replace('_', ' ')
    return base.replace('_', ' ')


def run_ingest(force: bool = False) -> dict:
    init_db()
    parser = PdfParser()

    already_ingested = set() if force else get_ingested_files()
    pdf_files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith('.pdf'))
    new_files = [f for f in pdf_files if f not in already_ingested]

    print(f"Found {len(pdf_files)} PDFs — {len(new_files)} to ingest "
          f"({'force mode' if force else 'incremental'})")

    results = {"processed": 0, "skipped": 0, "errors": []}

    for filename in new_files:
        path = os.path.join(CORPUS_DIR, filename)
        print(f"\n  [{results['processed'] + 1}/{len(new_files)}] {filename}")
        try:
            # 1. Parse PDF → text
            doc = parser.parse(path)
            text = doc.get("text", "").strip()
            if len(text) < 100:
                print(f"    ⚠ Skipping: text too short ({len(text)} chars)")
                results["skipped"] += 1
                continue

            doc_type   = _infer_doc_type(filename)
            source     = _infer_source(filename)
            title      = doc.get("title") or _infer_title(filename)
            published  = doc.get("published_at")

            # 2. NLP pipeline → SQLite (idempotent on SHA256 doc_id)
            pipeline_result = process_document(
                text=text,
                file_type="pdf",
                doc_type=doc_type,
                model="vader",
                title=title,
                source=source,
                published_at=published,
            )
            doc_id = pipeline_result["document"]["id"]
            n_entities = len(pipeline_result["signals"])

            # 3. Embed chunks → ChromaDB
            chunks = chunk_text(text)
            n_chunks = add_document_chunks(
                doc_id=doc_id,
                chunks=chunks,
                metadata={
                    "source":   source,
                    "doc_type": doc_type,
                    "title":    title,
                    "filename": filename,
                },
            )

            # 4. Mark as ingested
            mark_file_ingested(filename)
            results["processed"] += 1
            print(f"    ✅ {n_entities} entities, {len(chunks)} chunks "
                  f"({'added to ChromaDB' if n_chunks > 0 else 'ChromaDB already had it'})")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            results["errors"].append({"file": filename, "error": str(e)})

    print(f"\n{'='*50}")
    print(f"Ingest complete: {results['processed']} processed, "
          f"{results['skipped']} skipped, {len(results['errors'])} errors")
    if results["errors"]:
        for err in results["errors"]:
            print(f"  ❌ {err['file']}: {err['error']}")
    return results


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Ingest corpus PDFs into SQLite + ChromaDB")
    arg_parser.add_argument("--force", action="store_true",
                            help="Re-ingest all files (ignore corpus_ingested table)")
    args = arg_parser.parse_args()
    run_ingest(force=args.force)
```

- [ ] **Step 2 : Lancer l'ingest (première fois — ~2-5 min)**

```bash
cd /c/Users/ASUS/Desktop/github-project/sentiment_platform
python scripts/ingest_corpus.py
```

Résultat attendu : 14 fichiers traités, aucune erreur majeure. Certains PDFs peuvent avoir peu de texte extractible → "skipped" est normal.

- [ ] **Step 3 : Tester l'idempotence**

```bash
python scripts/ingest_corpus.py
```

Résultat attendu : `0 to ingest` — tous les fichiers sont déjà dans `corpus_ingested`.

- [ ] **Step 4 : Tester --force sur un fichier**

```bash
python scripts/ingest_corpus.py --force
```

Résultat attendu : les 14 fichiers sont re-traités, ChromaDB skip les chunks déjà présents (add_document_chunks retourne 0 si doc_id déjà présent).

- [ ] **Step 5 : Vérifier les résultats en DB**

```bash
python -c "
from shared.db.database import get_all_entities, get_ingested_files
entities = get_all_entities()
files = get_ingested_files()
print(f'Ingested files: {len(files)}')
print(f'Distinct entities in DB: {len(entities)}')
print(f'Sample entities: {entities[:10]}')
from module2_nlp.analysis.corpus_store import corpus_count
print(f'ChromaDB chunks: {corpus_count()}')
"
```

- [ ] **Step 6 : Commit**

```bash
git add scripts/ingest_corpus.py
git commit -m "feat: add idempotent corpus ingest script (PDF -> SQLite + ChromaDB)"
```

---

## Task 6 — Topic Model (BERTopic sur tweets)

**Files:**
- Create: `module2_nlp/analysis/topic_model.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/topic_model.py`**

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def fit_tweet_topics(texts: list) -> tuple:
    """
    Fit BERTopic on a list of tweet texts.
    Returns (topic_model, topics, topic_labels_dict)

    topic_labels_dict: {topic_id: human_readable_label}
    topic_id == -1 means outlier/unclassified.
    """
    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    topic_model = BERTopic(
        embedding_model=model,
        language="english",
        min_topic_size=5,
        nr_topics="auto",
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Build human-readable labels
    info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_labels[-1] = "Other"
        else:
            # Name from top keywords
            kws = topic_model.get_topic(tid)
            if kws:
                label = " / ".join([w for w, _ in kws[:3]])
                topic_labels[tid] = label
            else:
                topic_labels[tid] = f"Topic {tid}"

    return topic_model, topics, topic_labels


def get_topic_barchart(topic_model: BERTopic, top_n: int = 8):
    """Returns Plotly figure: top keywords per topic."""
    return topic_model.visualize_barchart(top_n_topics=top_n)


def get_topic_map(topic_model: BERTopic):
    """Returns Plotly figure: 2D topic map."""
    try:
        return topic_model.visualize_topics()
    except Exception:
        return None
```

- [ ] **Step 2 : Tester**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from module1_twitter.twitter.client import TwitterClient
from module2_nlp.analysis.topic_model import fit_tweet_topics

client = TwitterClient(backend='csv')
texts = [t['text'] for t in client.get_all()]
print(f'Fitting BERTopic on {len(texts)} tweets...')

topic_model, topics, labels = fit_tweet_topics(texts)
print(f'Topics found: {len(set(t for t in topics if t != -1))}')
print(f'Labels: {labels}')
assert len(labels) > 2, 'Expected at least 3 topics on 600 tweets'
print('✅ topic_model OK')
"
```

Résultat attendu : 4–10 topics (Oil/Iran, ECB, Geopolitics, China/PBOC…). Le fit prend ~10–30s selon CPU.

- [ ] **Step 3 : Commit**

```bash
git add module2_nlp/analysis/topic_model.py
git commit -m "feat: BERTopic wrapper for tweet topic modeling"
```

---

## Task 7 — Consensus & Divergence

**Files:**
- Create: `module2_nlp/analysis/consensus.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/consensus.py`**

```python
"""
Deux approches pour détecter consensus et divergences :
1. Marqueurs lexicaux (rapide, pour tweets individuels)
2. Variance inter-sources sur même entité (pour corpus + tweets agrégés)
"""
import numpy as np
from collections import defaultdict

CONSENSUS_MARKERS = [
    "confirms", "as expected", "in line with", "consensus",
    "widely expected", "broadly anticipated", "reaffirms",
    "consistent with", "unanimous", "all agree",
]

DIVERGENCE_MARKERS = [
    "contrary to", "surprisingly", "unexpectedly", "despite",
    "however", "risks to the upside", "risks to the downside",
    "at odds with", "challenges", "pushback", "diverges",
    "but wait", "yet markets", "contradicts",
]

WEAK_SIGNAL_MARKERS = [
    "first time since", "unusual", "historic", "unprecedented",
    "quietly", "under the radar", "few noticed", "overlooked",
    "little-noticed", "rarely discussed", "record",
]


def classify_tweet(text: str) -> dict:
    """
    Classifies a single tweet as consensus / divergence / signal_faible / neutral.
    Returns {label, consensus_score, divergence_score, signal_score}
    """
    text_lower = text.lower()
    c = sum(1 for m in CONSENSUS_MARKERS if m in text_lower)
    d = sum(1 for m in DIVERGENCE_MARKERS if m in text_lower)
    s = sum(1 for m in WEAK_SIGNAL_MARKERS if m in text_lower)

    if s > 0:
        label = "signal_faible"
    elif c > d:
        label = "consensus"
    elif d > c:
        label = "divergence"
    else:
        label = "neutral"

    return {"label": label, "consensus_score": c, "divergence_score": d, "signal_score": s}


def detect_divergences(entity_signals: list, threshold: float = 0.4) -> list:
    """
    entity_signals: [{"entity": str, "source": str, "score": float}, ...]
    Returns list of divergences sorted by delta (highest first).
    A divergence is when max_score - min_score > threshold across sources.

    Returns: [{entity, doc_a, score_a, doc_b, score_b, delta}, ...]
    """
    by_entity = defaultdict(list)
    for sig in entity_signals:
        by_entity[sig["entity"].lower()].append({
            "source": sig["source"],
            "score":  sig["score"],
        })

    divergences = []
    for entity, items in by_entity.items():
        if len(items) < 2:
            continue
        scores = [i["score"] for i in items]
        delta = max(scores) - min(scores)
        if delta > threshold:
            max_item = max(items, key=lambda x: x["score"])
            min_item = min(items, key=lambda x: x["score"])
            divergences.append({
                "entity":  entity,
                "doc_a":   max_item["source"],
                "score_a": round(max_item["score"], 3),
                "doc_b":   min_item["source"],
                "score_b": round(min_item["score"], 3),
                "delta":   round(delta, 3),
            })

    return sorted(divergences, key=lambda x: x["delta"], reverse=True)


def detect_consensus(entity_signals: list, min_sources: int = 2,
                     max_std: float = 0.20, min_abs_mean: float = 0.25) -> list:
    """
    Returns entities where multiple sources agree (low variance, strong signal).
    Returns: [{entity, mean_score, std, n_sources, direction}, ...]
    """
    by_entity = defaultdict(list)
    for sig in entity_signals:
        by_entity[sig["entity"].lower()].append(sig["score"])

    consensus = []
    for entity, scores in by_entity.items():
        if len(scores) < min_sources:
            continue
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std <= max_std and abs(mean) >= min_abs_mean:
            consensus.append({
                "entity":     entity,
                "mean_score": round(mean, 3),
                "std":        round(std, 3),
                "n_sources":  len(scores),
                "direction":  "bullish" if mean > 0 else "bearish",
            })

    return sorted(consensus, key=lambda x: abs(x["mean_score"]), reverse=True)
```

- [ ] **Step 2 : Tester**

```bash
python -c "
from module2_nlp.analysis.consensus import classify_tweet, detect_divergences, detect_consensus

# Lexical markers
r = classify_tweet('Surprisingly strong CPI data, contrary to market expectations')
assert r['label'] == 'divergence', f'Got: {r}'
r2 = classify_tweet('Data confirms consensus view on Fed trajectory')
assert r2['label'] == 'consensus', f'Got: {r2}'
print('✅ classify_tweet OK')

# Divergences
signals = [
    {'entity': 'oil', 'source': 'Goldman', 'score': 0.7},
    {'entity': 'oil', 'source': 'Natixis', 'score': -0.3},
    {'entity': 'gold', 'source': 'BofA', 'score': 0.5},
    {'entity': 'gold', 'source': 'SEB', 'score': 0.4},
]
divs = detect_divergences(signals, threshold=0.4)
assert len(divs) == 1
assert divs[0]['entity'] == 'oil'
assert divs[0]['delta'] == 1.0
print(f'✅ detect_divergences: {divs}')

# Consensus
cons = detect_consensus(signals, min_sources=2)
gold_cons = [c for c in cons if c['entity'] == 'gold']
assert len(gold_cons) == 1
assert gold_cons[0]['direction'] == 'bullish'
print(f'✅ detect_consensus: {cons}')
"
```

- [ ] **Step 3 : Commit**

```bash
git add module2_nlp/analysis/consensus.py
git commit -m "feat: consensus/divergence detection (lexical + inter-source variance)"
```

---

## Task 8 — Cross-source Alignment

**Files:**
- Create: `module2_nlp/analysis/cross_source.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/cross_source.py`**

```python
"""
Computes alignment between tweet sentiment and corpus stance on predefined themes.
Uses mean cosine similarity between tweet embeddings and corpus chunk embeddings.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_embed_model = None

# Predefined macro themes with keywords
THEMES = {
    "Oil / Energy":       ["oil", "crude", "WTI", "OPEC", "Brent", "petroleum", "Hormuz", "barrel"],
    "Gold / Safe Haven":  ["gold", "XAU", "bullion", "safe haven", "precious metal"],
    "Fed / US Rates":     ["Fed", "Federal Reserve", "rates", "inflation", "CPI", "hike", "cut", "hawkish", "dovish"],
    "Geopolitics":        ["Iran", "Russia", "Israel", "sanctions", "war", "ceasefire", "conflict", "Hormuz"],
    "Equities / Risk":    ["S&P", "SPX", "NASDAQ", "equities", "stocks", "recession", "risk"],
    "China / EM":         ["China", "PBOC", "GDP", "yuan", "Beijing", "emerging markets"],
    "ECB / Europe":       ["ECB", "European", "euro", "Germany", "EUR", "eurozone"],
}


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def compute_theme_alignment(theme_keywords: list, tweet_texts: list,
                             corpus_theme_emb) -> float | None:
    """
    Returns cosine similarity [0, 1] between tweet cluster and corpus passages on same theme.
    Returns None if not enough matching tweets (<3).
    """
    if corpus_theme_emb is None:
        return None

    model = _get_embed_model()
    matching = [t for t in tweet_texts
                if any(kw.lower() in t.lower() for kw in theme_keywords)]
    if len(matching) < 3:
        return None

    tweet_embs = model.encode(matching[:50], show_progress_bar=False)
    tweet_mean = tweet_embs.mean(axis=0)

    sim = cosine_similarity([tweet_mean], [corpus_theme_emb])[0][0]
    return float(np.clip(sim, 0, 1))


def get_all_theme_alignments(tweet_texts: list) -> dict:
    """
    Returns {theme_name: alignment_score} for all predefined themes.
    Scores missing if < 3 matching tweets or corpus empty.
    """
    from module2_nlp.analysis.corpus_store import get_corpus_theme_embedding

    results = {}
    for theme, keywords in THEMES.items():
        try:
            corpus_emb = get_corpus_theme_embedding(keywords)
            score = compute_theme_alignment(keywords, tweet_texts, corpus_emb)
            if score is not None:
                results[theme] = score
        except Exception:
            pass
    return results
```

- [ ] **Step 2 : Tester (nécessite corpus ingéré — après Task 5)**

```bash
python -c "
from module1_twitter.twitter.client import TwitterClient
from module2_nlp.analysis.cross_source import get_all_theme_alignments, THEMES

client = TwitterClient(backend='csv')
tweet_texts = [t['text'] for t in client.get_all()]

print('Computing cross-source alignments...')
alignments = get_all_theme_alignments(tweet_texts)
print(f'Themes computed: {len(alignments)}')
for theme, score in sorted(alignments.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(score * 20)
    print(f'  {theme:<25} {bar} {score:.2f}')
assert len(alignments) > 0, 'Need at least one theme with data'
print('✅ cross_source OK')
"
```

- [ ] **Step 3 : Commit**

```bash
git add module2_nlp/analysis/cross_source.py
git commit -m "feat: cross-source tweet vs corpus alignment by theme"
```

---

## Task 9 — Dashboard Tab 1 : Macro Digest

**Files:**
- Modify: `dashboard/app.py` (réécriture complète, structurée en fonctions par tab)

- [ ] **Step 1 : Réécrire `dashboard/app.py` — structure + Tab 1**

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from config import ASSETS, TRACKED_ENTITIES, ALERT_THRESHOLD

st.set_page_config(
    page_title="Macro Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES & DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_vader():
    from shared.nlp.vader_sentiment import VaderSentiment
    return VaderSentiment()

@st.cache_resource
def get_finbert():
    from shared.nlp.finbert_sentiment import FinBERTSentiment
    return FinBERTSentiment()

def get_analyzer(model_name: str):
    return get_finbert() if model_name == "FinBERT" else get_vader()


@st.cache_data(ttl=3600, show_spinner="Loading & analysing tweets...")
def load_enriched_tweets(model_name: str = "VADER") -> pd.DataFrame:
    """Load CSV tweets, run sentiment + consensus classification."""
    from module1_twitter.twitter.client import TwitterClient
    from module1_twitter.nlp.preprocessor import clean_tweet
    from module2_nlp.analysis.consensus import classify_tweet

    client = TwitterClient(backend="csv")
    tweets = client.get_all()
    analyzer = get_analyzer(model_name)

    rows = []
    for t in tweets:
        clean = clean_tweet(t["text"])
        result = analyzer.analyze(clean)
        meta = classify_tweet(t["text"])
        rows.append({
            "id":              t["id"],
            "created_at":      t["created_at"],
            "author":          t["author"],
            "text":            t["text"],
            "text_clean":      clean,
            "sentiment_score": result["score"],
            "sentiment_label": result["label"],
            "view_type":       meta["label"],
        })
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=3600, show_spinner="Fitting topic model...")
def load_tweet_topics(model_name: str = "VADER"):
    """Fit BERTopic on tweets. Returns (topic_model, topics_list, labels_dict)."""
    from module2_nlp.analysis.topic_model import fit_tweet_topics
    df = load_enriched_tweets(model_name)
    texts = df["text"].tolist()
    try:
        topic_model, topics, labels = fit_tweet_topics(texts)
        return topic_model, topics, labels
    except Exception as e:
        st.warning(f"BERTopic skipped: {e}")
        return None, [-1] * len(texts), {-1: "All"}


@st.cache_data(ttl=3600)
def load_corpus_entity_signals() -> pd.DataFrame:
    """Load entity sentiment signals from SQLite corpus ingest."""
    from shared.db.database import get_session, EntitySentiment, Document

    session = get_session()
    try:
        rows = (
            session.query(EntitySentiment, Document)
            .join(Document, EntitySentiment.document_id == Document.id)
            .all()
        )
        records = []
        for es, doc in rows:
            source_label = (doc.source or doc.title or "unknown")[:30]
            records.append({
                "entity":  es.entity.lower(),
                "source":  source_label,
                "score":   es.sentiment_score,
                "doc_type": doc.doc_type or "unknown",
            })
        return pd.DataFrame(records) if records else pd.DataFrame(columns=["entity","source","score","doc_type"])
    finally:
        session.close()


def build_entity_heatmap(tweets_df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds entity × source pivot table.
    Sources = document source labels + 'Tweets'.
    """
    all_keywords = {kw.lower() for kws in TRACKED_ENTITIES.values() for kw in kws}

    # Tweet signals: mean sentiment for tweets containing each keyword
    tweet_records = []
    for kw in all_keywords:
        mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
        matching = tweets_df[mask]
        if len(matching) >= 3:
            tweet_records.append({
                "entity": kw,
                "source": "📡 Tweets",
                "score":  matching["sentiment_score"].mean(),
            })

    tweet_df = pd.DataFrame(tweet_records)

    # Combine
    if corpus_df.empty and tweet_df.empty:
        return pd.DataFrame()

    parts = []
    if not corpus_df.empty:
        parts.append(corpus_df[["entity", "source", "score"]])
    if not tweet_df.empty:
        parts.append(tweet_df)

    combined = pd.concat(parts, ignore_index=True)
    pivot = (
        combined.groupby(["entity", "source"])["score"]
        .mean()
        .reset_index()
        .pivot(index="entity", columns="source", values="score")
    )
    # Move Tweets column last
    cols = [c for c in pivot.columns if c != "📡 Tweets"] + (["📡 Tweets"] if "📡 Tweets" in pivot.columns else [])
    return pivot[cols]


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🧭 Macro Digest",
    "📡 Tweet Intelligence",
    "📚 Corpus Analysis",
    "📊 Backtest",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MACRO DIGEST
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("🧭 Macro Digest")
    st.caption("Combined view: FinancialJuice tweets (4 days) + 14 macro documents")

    model_t1 = st.sidebar.radio("Sentiment Model", ["VADER", "FinBERT"], key="t1_model")

    try:
        tweets_df = load_enriched_tweets(model_t1)
        corpus_df = load_corpus_entity_signals()

        # ── KPI Row ──────────────────────────────────────────────────────────
        from shared.db.database import get_ingested_files
        n_docs = len(get_ingested_files())

        from module2_nlp.analysis.consensus import detect_divergences, detect_consensus
        entity_signals = []
        if not corpus_df.empty:
            for _, row in corpus_df.iterrows():
                entity_signals.append({"entity": row["entity"], "source": row["source"], "score": row["score"]})

        # Add tweet signals to entity_signals
        all_keywords = {kw.lower() for kws in TRACKED_ENTITIES.values() for kw in kws}
        for kw in all_keywords:
            mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
            matching = tweets_df[mask]
            if len(matching) >= 3:
                entity_signals.append({"entity": kw, "source": "📡 Tweets", "score": matching["sentiment_score"].mean()})

        divergences = detect_divergences(entity_signals, threshold=0.4)
        consensus_list = detect_consensus(entity_signals, min_sources=2)

        topic_model, topics, labels = load_tweet_topics(model_t1)
        n_topics = len(set(t for t in topics if t != -1))

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Tweets analysés", len(tweets_df))
        k2.metric("Docs ingérés", n_docs)
        k3.metric("Topics détectés", n_topics)
        k4.metric("Consensus", len(consensus_list))
        k5.metric("Divergences", len(divergences))

        st.divider()

        # ── Heatmap + Digest ─────────────────────────────────────────────────
        col_heat, col_digest = st.columns([3, 2])

        with col_heat:
            st.subheader("Entity × Source Heatmap")
            heatmap_pivot = build_entity_heatmap(tweets_df, corpus_df)
            if heatmap_pivot.empty:
                st.info("No entity data yet. Run: python scripts/ingest_corpus.py")
            else:
                fig_heat = px.imshow(
                    heatmap_pivot,
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    zmin=-1, zmax=1,
                    aspect="auto",
                    title="Sentiment score per entity & source",
                )
                fig_heat.update_layout(height=500, margin=dict(t=40, b=10))
                fig_heat.update_xaxes(tickangle=45)
                st.plotly_chart(fig_heat, use_container_width=True)

        with col_digest:
            st.subheader("📋 Macro Digest")

            st.markdown("**📊 Consensus Views**")
            if consensus_list:
                for c in consensus_list[:3]:
                    icon = "🟢" if c["direction"] == "bullish" else "🔴"
                    st.info(
                        f"{icon} **{c['entity'].upper()}** — {c['direction'].capitalize()} "
                        f"({c['n_sources']} sources, avg {c['mean_score']:+.2f})"
                    )
            else:
                st.info("No strong consensus detected yet.")

            st.markdown("**⚡ Divergences**")
            if divergences:
                for d in divergences[:3]:
                    st.warning(
                        f"**{d['entity'].upper()}** — "
                        f"{d['doc_a']} ({d['score_a']:+.2f}) vs "
                        f"{d['doc_b']} ({d['score_b']:+.2f}) · Δ={d['delta']:.2f}"
                    )
            else:
                st.warning("No major divergences detected.")

            # Weak signals: themes in corpus absent from tweets
            st.markdown("**🔍 Weak Signals**")
            if not corpus_df.empty:
                corpus_entities = set(corpus_df["entity"].unique())
                tweet_keywords = set()
                for kw in all_keywords:
                    mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
                    if mask.sum() >= 3:
                        tweet_keywords.add(kw)
                weak = corpus_entities - tweet_keywords
                if weak:
                    for w in list(weak)[:2]:
                        st.success(f"🔍 **{w}** — mentioned in corpus, low volume in tweets")
                else:
                    st.success("No obvious weak signals detected.")
            else:
                st.success("Run ingest to detect weak signals.")

        st.divider()

        # ── Cross-source Alignment ───────────────────────────────────────────
        st.subheader("Cross-Source Alignment: Tweets ↔ Corpus (by Theme)")
        try:
            from module2_nlp.analysis.cross_source import get_all_theme_alignments
            tweet_texts = tweets_df["text"].tolist()
            with st.spinner("Computing cross-source alignments..."):
                alignments = get_all_theme_alignments(tweet_texts)

            if alignments:
                align_df = pd.DataFrame([
                    {"Theme": k, "Alignment": v, "Status": "Aligned" if v > 0.6 else ("Mixed" if v > 0.4 else "Divergent")}
                    for k, v in sorted(alignments.items(), key=lambda x: x[1], reverse=True)
                ])
                color_map = {"Aligned": "#22c55e", "Mixed": "#f59e0b", "Divergent": "#ef4444"}
                fig_align = px.bar(
                    align_df, x="Alignment", y="Theme", orientation="h",
                    color="Status",
                    color_discrete_map=color_map,
                    title="Similarity between tweet stance and corpus on each macro theme",
                    range_x=[0, 1],
                )
                fig_align.add_vline(x=0.6, line_dash="dash", line_color="green", annotation_text="Aligned")
                fig_align.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Divergent")
                fig_align.update_layout(height=350, margin=dict(t=40, b=10))
                st.plotly_chart(fig_align, use_container_width=True)
            else:
                st.info("Not enough data for cross-source alignment. Ensure corpus is ingested.")
        except Exception as e:
            st.error(f"Cross-source alignment error: {e}")

    except Exception as e:
        st.error(f"Macro Digest error: {e}")
        st.exception(e)
```

- [ ] **Step 2 : Smoke test Tab 1**

```bash
streamlit run dashboard/app.py --server.headless true &
sleep 10
curl -sf http://localhost:8501/_stcore/health && echo "✅ Dashboard UP" || echo "❌ FAILED"
kill %1 2>/dev/null
```

- [ ] **Step 3 : Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard Tab 1 Macro Digest (heatmap + digest + cross-source)"
```

---

## Task 10 — Dashboard Tab 2 : Tweet Intelligence

**Files:**
- Modify: `dashboard/app.py` — ajouter le bloc `with tab2:`

- [ ] **Step 1 : Ajouter Tab 2 dans `dashboard/app.py`** (après le bloc `with tab1:`)

```python
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TWEET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("📡 Tweet Intelligence")

    model_t2 = st.sidebar.radio("Model (Tab 2)", ["VADER", "FinBERT"], key="t2_model")

    try:
        tweets_df = load_enriched_tweets(model_t2)
        topic_model, topics, labels = load_tweet_topics(model_t2)

        # Attach topic info to df
        df2 = tweets_df.copy()
        df2["topic_id"] = topics
        df2["topic_label"] = df2["topic_id"].map(labels).fillna("Other")

        # ── Sidebar filters ───────────────────────────────────────────────────
        unique_topics = sorted(df2["topic_label"].unique())
        sel_topics = st.sidebar.multiselect("Topics", unique_topics, default=unique_topics[:], key="t2_topics")
        sel_sentiment = st.sidebar.multiselect(
            "Sentiment", ["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"], key="t2_sent"
        )
        sel_type = st.sidebar.multiselect(
            "Type", ["consensus", "divergence", "signal_faible", "neutral"],
            default=["consensus", "divergence", "signal_faible", "neutral"], key="t2_type"
        )

        # Filter
        mask = (
            df2["topic_label"].isin(sel_topics) &
            df2["sentiment_label"].isin(sel_sentiment) &
            df2["view_type"].isin(sel_type)
        )
        filtered = df2[mask].copy()

        # ── Timeline ─────────────────────────────────────────────────────────
        st.subheader(f"Sentiment Timeline — {len(filtered)} tweets")
        timeline = (
            filtered.groupby([filtered["created_at"].dt.floor("H"), "sentiment_label"])
            .size()
            .reset_index(name="count")
        )
        timeline.columns = ["hour", "sentiment", "count"]
        if not timeline.empty:
            fig_time = px.line(
                timeline, x="hour", y="count", color="sentiment",
                color_discrete_map={
                    "positive": "#22c55e",
                    "negative": "#ef4444",
                    "neutral":  "#94a3b8",
                },
                markers=True,
            )
            fig_time.update_layout(height=280, margin=dict(t=20, b=10))
            st.plotly_chart(fig_time, use_container_width=True)

        st.divider()

        # ── Feed + Topic Map ──────────────────────────────────────────────────
        col_feed, col_map = st.columns([1, 1])

        SENT_ICON = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        TYPE_BADGE = {"consensus": "📊", "divergence": "⚡", "signal_faible": "🔍", "neutral": ""}

        with col_feed:
            st.subheader("Interactive Feed")
            page_size = 50
            page = st.number_input("Page", min_value=1,
                                   max_value=max(1, len(filtered) // page_size + 1),
                                   value=1, key="t2_page")
            start = (page - 1) * page_size
            page_df = filtered.iloc[start:start + page_size]

            for _, row in page_df.iterrows():
                icon  = SENT_ICON.get(row["sentiment_label"], "⚪")
                badge = TYPE_BADGE.get(row["view_type"], "")
                ts    = row["created_at"].strftime("%b %d %H:%M") if pd.notna(row["created_at"]) else ""
                st.markdown(
                    f"{icon} {badge} {row['text']}\n\n"
                    f"<small style='color:gray'>{ts} · score: {row['sentiment_score']:+.2f} · {row['topic_label']}</small>",
                    unsafe_allow_html=True,
                )
                st.divider()

        with col_map:
            st.subheader("Topic Map")
            if topic_model is not None:
                try:
                    fig_map = topic_model.visualize_topics()
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.info(f"Topic map unavailable: {e}")
            else:
                st.info("Topic model not available.")

            # Keywords bar chart
            st.subheader("Top Keywords by Topic")
            if topic_model is not None:
                try:
                    fig_bar = topic_model.visualize_barchart(top_n_topics=6)
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.info(f"Barchart unavailable: {e}")

    except Exception as e:
        st.error(f"Tweet Intelligence error: {e}")
        st.exception(e)
```

- [ ] **Step 2 : Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard Tab 2 Tweet Intelligence (feed + topic map)"
```

---

## Task 11 — Dashboard Tab 3 : Corpus Analysis

**Files:**
- Modify: `dashboard/app.py` — ajouter le bloc `with tab3:`

- [ ] **Step 1 : Ajouter Tab 3 dans `dashboard/app.py`**

```python
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORPUS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📚 Corpus Analysis")

    # Sidebar controls
    doc_model = st.sidebar.radio("Model (Tab 3)", ["VADER", "FinBERT"], key="t3_model")
    doc_type_filter = st.sidebar.selectbox(
        "Doc type filter",
        ["all", "research_note", "news", "hedge_fund_letter", "fomc_minutes", "earnings_call"],
        key="t3_dtype"
    )

    # Re-ingest button
    if st.sidebar.button("🔄 Re-ingest corpus", key="t3_reingest"):
        with st.spinner("Running ingest_corpus.py..."):
            try:
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "scripts/ingest_corpus.py"],
                    capture_output=True, text=True,
                    cwd=os.path.join(os.path.dirname(__file__), '..')
                )
                if result.returncode == 0:
                    st.sidebar.success("Ingest complete!")
                    st.cache_data.clear()
                else:
                    st.sidebar.error(f"Ingest error: {result.stderr[:200]}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    try:
        corpus_df = load_corpus_entity_signals()

        if corpus_df.empty:
            st.warning("No corpus data. Run: `python scripts/ingest_corpus.py`")
        else:
            if doc_type_filter != "all":
                corpus_df = corpus_df[corpus_df["doc_type"] == doc_type_filter]

            # ── Heatmap Documents × Entities ─────────────────────────────────
            st.subheader("Document × Entity Sentiment Heatmap")
            pivot_doc = (
                corpus_df.groupby(["source", "entity"])["score"]
                .mean()
                .reset_index()
                .pivot(index="source", columns="entity", values="score")
            )

            # Keep only entities with ≥2 mentions across sources
            entity_counts = corpus_df.groupby("entity")["source"].nunique()
            keep_entities = entity_counts[entity_counts >= 2].index
            pivot_doc = pivot_doc[[c for c in pivot_doc.columns if c in keep_entities]]

            if not pivot_doc.empty:
                fig_doc_heat = px.imshow(
                    pivot_doc,
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    zmin=-1, zmax=1,
                    aspect="auto",
                    title="Mean sentiment per document × entity",
                )
                fig_doc_heat.update_layout(height=450, margin=dict(t=40, b=10))
                fig_doc_heat.update_xaxes(tickangle=45)
                st.plotly_chart(fig_doc_heat, use_container_width=True)

            st.divider()

            # ── Predefined Themes + Emerging Signals ─────────────────────────
            from module2_nlp.analysis.cross_source import THEMES
            col_themes, col_emerging = st.columns(2)

            with col_themes:
                st.subheader("📌 Predefined Themes")
                theme_scores = {}
                for theme, keywords in THEMES.items():
                    mask = corpus_df["entity"].apply(
                        lambda e: any(kw.lower() in e for kw in keywords)
                    )
                    subset = corpus_df[mask]
                    if len(subset) > 0:
                        theme_scores[theme] = subset["score"].mean()

                if theme_scores:
                    ts_df = pd.DataFrame([
                        {"Theme": k, "Avg Sentiment": v}
                        for k, v in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
                    ])
                    fig_themes = px.bar(
                        ts_df, x="Avg Sentiment", y="Theme",
                        orientation="h",
                        color="Avg Sentiment",
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        range_x=[-1, 1],
                    )
                    fig_themes.update_layout(height=350, margin=dict(t=20, b=10))
                    st.plotly_chart(fig_themes, use_container_width=True)

            with col_emerging:
                st.subheader("🔍 Emerging Signals (TF-IDF)")
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from shared.db.database import get_session, Document

                    session = get_session()
                    docs_text = [d.title or "" for d in session.query(Document).all()]
                    session.close()

                    known_kws = {kw.lower() for kws in THEMES.values() for kw in kws}
                    if len(docs_text) >= 3:
                        vec = TfidfVectorizer(stop_words="english", max_features=100, ngram_range=(1, 2))
                        vec.fit(docs_text)
                        tfidf_terms = vec.get_feature_names_out()
                        emerging = [t for t in tfidf_terms if not any(k in t for k in known_kws)][:10]

                        if emerging:
                            fig_emerg = px.bar(
                                pd.DataFrame({"Term": emerging, "Rank": range(1, len(emerging)+1)}),
                                x="Rank", y="Term", orientation="h",
                                title="Terms not in predefined taxonomy",
                            )
                            fig_emerg.update_layout(height=350, margin=dict(t=40, b=10))
                            st.plotly_chart(fig_emerg, use_container_width=True)
                        else:
                            st.info("No emerging terms found.")
                    else:
                        st.info("Not enough documents for TF-IDF.")
                except Exception as e:
                    st.info(f"TF-IDF unavailable: {e}")

            st.divider()

            # ── Divergence Table ──────────────────────────────────────────────
            st.subheader("⚡ Inter-Document Divergences")
            from module2_nlp.analysis.consensus import detect_divergences

            div_signals = [
                {"entity": row["entity"], "source": row["source"], "score": row["score"]}
                for _, row in corpus_df.iterrows()
            ]
            divs = detect_divergences(div_signals, threshold=0.35)

            if divs:
                div_df = pd.DataFrame(divs)
                div_df.columns = ["Entity", "Bullish Source", "Score +", "Bearish Source", "Score -", "Δ"]
                st.dataframe(
                    div_df.style.background_gradient(subset=["Δ"], cmap="Reds"),
                    use_container_width=True,
                )
            else:
                st.info("No significant divergences (threshold Δ > 0.35).")

            st.divider()

            # ── Semantic Search ───────────────────────────────────────────────
            st.subheader("🔎 Semantic Search on Corpus")
            query = st.text_input("Search the 14 documents...", placeholder="e.g. Fed rate trajectory 2026", key="t3_query")
            if query:
                from module2_nlp.analysis.corpus_store import semantic_search
                with st.spinner("Searching..."):
                    results = semantic_search(query, n_results=5)
                if results:
                    for r in results:
                        with st.expander(f"📄 {r['source']} ({r['doc_type']}) — similarity: {r['similarity']:.2f}"):
                            st.markdown(r["text"])
                else:
                    st.info("No results. Is the corpus ingested?")

    except Exception as e:
        st.error(f"Corpus Analysis error: {e}")
        st.exception(e)
```

- [ ] **Step 2 : Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard Tab 3 Corpus Analysis (heatmap + themes + divergences + search)"
```

---

## Task 12 — Dashboard Tab 4 : Backtest

**Files:**
- Modify: `dashboard/app.py` — ajouter le bloc `with tab4:`

- [ ] **Step 1 : Ajouter Tab 4 dans `dashboard/app.py`**

```python
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📊 Backtest")

    source_bt = st.radio("Signal source", ["Tweets CSV", "Corpus Documents"], horizontal=True, key="bt_source")
    bt_asset  = st.selectbox("Price asset", list(ASSETS.keys()) + ["oil", "gold", "EUR/USD"], key="bt_asset")

    horizons_map = {"15min": 0.25, "30min": 0.5, "1h": 1, "4h": 4, "12h": 12, "1d": 24, "3d": 72, "1w": 168}
    horizon_lbl  = st.select_slider("Horizon", options=list(horizons_map.keys()), value="1d", key="bt_horizon")
    horizon_hrs  = horizons_map[horizon_lbl]

    if horizon_hrs < 1:
        st.info("📡 Source: Alpha Vantage 1min")
    elif horizon_hrs <= 24:
        st.info("📈 Source: yfinance 1h")
    else:
        st.info("📈 Source: yfinance 1d")

    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("Start", value=datetime(2024, 1, 1), key="bt_start")
    with col2:
        bt_end = st.date_input("End", value=datetime(2025, 1, 1), key="bt_end")

    if source_bt == "Corpus Documents":
        from shared.db.database import get_all_entities
        entities = get_all_entities()
        if entities:
            bt_entity = st.selectbox("Entity", entities, key="bt_entity")
        else:
            st.info("No corpus entities yet. Run ingest first.")
            bt_entity = None
    else:
        bt_entity = None

    if st.button("▶ Run Backtest", key="bt_run"):
        try:
            with st.spinner("Running backtest..."):
                from shared.backtest.engine import BacktestEngine

                if source_bt == "Tweets CSV":
                    df_bt = load_enriched_tweets("VADER")
                    signals = [
                        {"timestamp": row["created_at"], "signal_value": row["sentiment_score"]}
                        for _, row in df_bt.iterrows()
                    ]
                else:
                    if not bt_entity:
                        st.warning("Select an entity first.")
                        st.stop()
                    from shared.db.database import get_doc_signals
                    raw = get_doc_signals(entity=bt_entity)
                    signals = [
                        {"timestamp": s.get("published_at", datetime(2024, 6, 1)),
                         "signal_value": s["signal_value"]}
                        for s in raw if s.get("signal_value") is not None
                    ]

                if not signals:
                    st.warning("No signals found.")
                else:
                    engine = BacktestEngine()
                    results = engine.run(signals, bt_asset, horizon_hrs)

                    if "error" in results:
                        st.warning(results["error"])
                    else:
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("Dir. Accuracy", f"{results['directional_accuracy']:.2%}")
                        k2.metric("Pearson r",     f"{results['pearson_r']:.3f}")
                        k3.metric("Spearman r",    f"{results['spearman_r']:.3f}")
                        k4.metric("N Signals",     results["n_signals"])
                        k5.metric("p-value",       f"{results['pearson_pval']:.4f}")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            buckets = results["forward_returns_by_bucket"]
                            fig_b = go.Figure(go.Bar(
                                x=list(buckets.keys()), y=list(buckets.values()),
                                marker_color=["red" if v < 0 else "green" for v in buckets.values()]
                            ))
                            fig_b.update_layout(title="Forward Returns by Bucket", height=320)
                            st.plotly_chart(fig_b, use_container_width=True)

                        with col_b:
                            raw_df = results.get("raw_df")
                            if raw_df is not None and len(raw_df) > 1:
                                fig_s = px.scatter(
                                    raw_df, x="signal", y="fwd_return",
                                    trendline="ols",
                                    title="Signal vs Forward Return",
                                )
                                fig_s.update_layout(height=320)
                                st.plotly_chart(fig_s, use_container_width=True)

                        rc = results.get("rolling_correlation", [])
                        if rc:
                            rc_df = pd.DataFrame(rc)
                            fig_rc = go.Figure()
                            fig_rc.add_trace(go.Scatter(x=rc_df["timestamp"], y=rc_df["rolling_corr"], mode="lines"))
                            fig_rc.add_hline(y=0, line_dash="dash")
                            fig_rc.update_layout(title="Rolling Correlation (20 obs)", height=280)
                            st.plotly_chart(fig_rc, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.exception(e)
```

- [ ] **Step 2 : Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard Tab 4 Backtest (tweets + corpus, merged)"
```

---

## Task 13 — Livrables finaux : README + DOCUMENTATION.md

**Files:**
- Modify: `README.md` — marquer toutes les features ✅ DONE
- Modify: `DOCUMENTATION.md` — retirer mock sections, documenter nouveaux modules

- [ ] **Step 1 : Mettre à jour `README.md`** — passer toutes les features "⬜ TODO" en "✅ DONE" (ou "⚠ PARTIAL" si non terminé)

- [ ] **Step 2 : Mettre à jour `DOCUMENTATION.md`**

Supprimer ou marquer "deprecated" les sections :
- `mock_backend.py`
- `mock_documents.py`

Ajouter les sections :
- `csv_backend.py` — interface, colonnes CSV attendues
- `module2_nlp/analysis/` — corpus_store, consensus, cross_source, topic_model
- `scripts/ingest_corpus.py` — usage, idempotence, --force flag

- [ ] **Step 3 : Smoke test final**

```bash
streamlit run dashboard/app.py --server.headless true &
sleep 12
curl -sf http://localhost:8501/_stcore/health && echo "✅ Dashboard UP" || echo "❌ FAILED"
kill %1 2>/dev/null
```

- [ ] **Step 4 : Commit final**

```bash
git add README.md DOCUMENTATION.md
git commit -m "docs: final README and DOCUMENTATION update — all features complete"
```

---

## Self-review

**Spec coverage check :**
- ✅ CSV backend → Task 3
- ✅ Corpus ingest (ChromaDB + SQLite, idempotent) → Tasks 4–5
- ✅ BERTopic → Task 6
- ✅ Consensus/divergence → Task 7
- ✅ Cross-source alignment → Task 8
- ✅ Tab 1 Macro Digest (5 KPIs, heatmap, digest, cross-source) → Task 9
- ✅ Tab 2 Tweet Intelligence (timeline, feed, topic map) → Task 10
- ✅ Tab 3 Corpus (heatmap docs×entities, themes, divergences, search) → Task 11
- ✅ Tab 4 Backtest (source radio, merged) → Task 12
- ✅ notes_choix_techniques.md → Task 0
- ✅ README de lancement → Tasks 0 + 13
- ✅ DOCUMENTATION.md → Tasks 0 + 13
- ✅ CLAUDE.md mis à jour → Task 0

**Type consistency check :**
- `load_enriched_tweets()` → `pd.DataFrame` avec colonnes : id, created_at, author, text, text_clean, sentiment_score, sentiment_label, view_type — cohérent Tasks 9/10/12
- `load_corpus_entity_signals()` → `pd.DataFrame` avec colonnes : entity, source, score, doc_type — cohérent Tasks 9/11
- `detect_divergences(entity_signals)` attend `[{entity, source, score}]` — cohérent Tasks 9/11
- `detect_consensus(entity_signals)` attend `[{entity, source, score}]` — cohérent Tasks 9/11
- `add_document_chunks(doc_id, chunks, metadata)` → `int` — cohérent Tasks 4/5
- `semantic_search(query, n_results)` → `[{text, source, doc_type, title, filename, similarity}]` — cohérent Tasks 4/11

**Placeholder scan :** aucun TBD/TODO dans le code. Tous les blocs de code sont complets.
