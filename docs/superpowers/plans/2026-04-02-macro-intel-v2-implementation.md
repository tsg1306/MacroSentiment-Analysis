# Macro Intelligence Dashboard v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implémenter le redesign v2 du dashboard : classification des documents par domaine/stance/trade, résumés 10 bullets extractifs, et 4 tabs redessinés.

**Architecture:** Nouveaux modules d'analyse (document_classifier + summarizer) alimentent l'ingest script qui enrichit SQLite. Le dashboard lit tout depuis la DB/cache — jamais de traitement lourd à la demande.

**Tech Stack:** sklearn TF-IDF, regex/lexical markers, sentence-transformers, ChromaDB, BERTopic+KMeans, Streamlit, Plotly, SQLAlchemy/SQLite.

**État de départ :** Tasks 0-3, 6-8 du plan v1 sont ✅ DONE. Tasks 4-5 (ChromaDB + ingest) restent à faire. La DB a déjà la table `corpus_ingested`.

---

## Fichiers créés / modifiés

| Fichier | Action |
|---------|--------|
| `shared/db/database.py` | Modifié — ajout `DocumentAnalysis` table + CRUD |
| `module2_nlp/analysis/document_classifier.py` | Créé |
| `module2_nlp/analysis/summarizer.py` | Créé |
| `module2_nlp/analysis/corpus_store.py` | Créé (depuis plan v1) |
| `scripts/ingest_corpus.py` | Créé + intègre classifier + summarizer |
| `dashboard/app.py` | Réécrit complet (4 tabs v2) |
| `CLAUDE.md` | Mis à jour statut tâches |

---

## Task A — DB : table document_analysis

**Files:**
- Modify: `shared/db/database.py`

- [ ] **Step 1 : Ajouter la classe `DocumentAnalysis` après `CorpusIngested`**

```python
import json as _json

class DocumentAnalysis(Base):
    __tablename__ = "document_analysis"
    doc_id         = Column(Text, ForeignKey("documents.id"), primary_key=True)
    domains        = Column(Text)   # JSON list ex: '["Macro / Rates", "Oil / Energy"]'
    primary_domain = Column(Text)
    stance         = Column(Text)   # institutional | investor | research_note
    trade_signal   = Column(Text)   # explicit | implicit | none
    explicit_count = Column(Integer, default=0)
    implicit_count = Column(Integer, default=0)
    summary_json   = Column(Text)   # JSON list of {text, type}
```

- [ ] **Step 2 : Ajouter les fonctions CRUD à la fin de `database.py`**

```python
def save_document_analysis(record: dict):
    """Upsert document analysis record."""
    session = get_session()
    try:
        existing = session.get(DocumentAnalysis, record["doc_id"])
        if existing is None:
            session.add(DocumentAnalysis(**record))
        else:
            for k, v in record.items():
                setattr(existing, k, v)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_document_analysis(doc_id: str) -> dict | None:
    session = get_session()
    try:
        row = session.get(DocumentAnalysis, doc_id)
        if row is None:
            return None
        return {c.name: getattr(row, c.name) for c in DocumentAnalysis.__table__.columns}
    finally:
        session.close()


def get_all_document_analyses(
    domain_filter: list = None,
    stance_filter: str = None,
    trade_filter: str = None,
) -> list:
    """Returns all document analyses with optional filters."""
    import json
    session = get_session()
    try:
        q = session.query(DocumentAnalysis, Document).join(
            Document, DocumentAnalysis.doc_id == Document.id
        )
        rows = q.all()
        results = []
        for da, doc in rows:
            domains = json.loads(da.domains or "[]")
            if domain_filter and not any(d in domains for d in domain_filter):
                continue
            if stance_filter and da.stance != stance_filter:
                continue
            if trade_filter and da.trade_signal != trade_filter:
                continue
            results.append({
                "doc_id":        da.doc_id,
                "title":         doc.title,
                "source":        doc.source,
                "doc_type":      doc.doc_type,
                "published_at":  doc.published_at,
                "domains":       domains,
                "primary_domain": da.primary_domain,
                "stance":        da.stance,
                "trade_signal":  da.trade_signal,
                "explicit_count": da.explicit_count,
                "implicit_count": da.implicit_count,
                "summary_json":  da.summary_json,
            })
        return results
    finally:
        session.close()
```

- [ ] **Step 3 : Tester**

```bash
python -c "
from shared.db.database import init_db, save_document_analysis, get_document_analysis, get_all_document_analyses
import json
init_db()
save_document_analysis({
    'doc_id': 'test123',
    'domains': json.dumps(['Macro / Rates', 'Oil / Energy']),
    'primary_domain': 'Oil / Energy',
    'stance': 'research_note',
    'trade_signal': 'explicit',
    'explicit_count': 2,
    'implicit_count': 1,
    'summary_json': json.dumps([{'text': 'Buy WTI', 'type': 'explicit'}]),
})
r = get_document_analysis('test123')
assert r['trade_signal'] == 'explicit'
all_r = get_all_document_analyses(trade_filter='explicit')
assert any(x['doc_id'] == 'test123' for x in all_r)
print('Task A OK')
"
```

- [ ] **Step 4 : Commit**

```bash
git add shared/db/database.py
git commit -m "feat: add document_analysis table with domain/stance/trade/summary fields"
```

---

## Task B — Document Classifier

**Files:**
- Create: `module2_nlp/analysis/document_classifier.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/document_classifier.py`**

```python
"""
Classifies a document on three axes:
1. Domain themes (TF-IDF keyword matching, multi-label, 7 themes)
2. Stance (institutional vs investor vs research_note)
3. Trade signal presence (explicit / implicit / none)
"""
import re

THEMES = {
    "Macro / Rates":   ["inflation","gdp","rates","central bank","monetary","fed","ecb",
                        "hike","cut","treasury","yield","fomc","tapering","tightening"],
    "Oil / Energy":    ["oil","crude","opec","gas","barrel","refinery","hormuz","wti",
                        "brent","petroleum","energy","lng","pipeline","supply cut"],
    "Geopolitics":     ["war","sanctions","conflict","iran","israel","russia","nuclear",
                        "military","ceasefire","strike","attack","escalation","nato"],
    "Equities / Risk": ["equity","stocks","s&p","earnings","recession","pe ratio",
                        "buyback","dividend","nasdaq","market rally","risk-off","risk-on"],
    "China / EM":      ["china","pboc","yuan","emerging markets","beijing","renminbi",
                        "chinese","trade war","tariff","EM","developing"],
    "Europe / FX":     ["euro","ecb","germany","eurozone","eur","europe","european",
                        "draghi","lagarde","bund","periphery"],
}

INSTITUTIONAL_MARKERS = [
    "our analysts","we forecast","base case","our estimate","we expect",
    "consensus","client note","our model","survey says","analyst consensus",
    "house view","according to our","in our view",
]

INVESTOR_MARKERS = [
    "i am long","we are buying","our position","we hold","added to",
    "trimmed","initiated position","we own","personal view","i believe",
    "our fund","portfolio position","we remain long","we remain short",
]

EXPLICIT_TRADE_MARKERS = [
    "buy ","sell ","overweight","underweight"," long ","short ",
    "target price","price target"," tp ","upgrade","downgrade",
    "add to position","trim position","initiate","reiterate buy",
    "reiterate sell","strong buy","strong sell",
]

IMPLICIT_TRADE_MARKERS = [
    "attractive","compelling","opportunity","well-positioned",
    "upside","downside risk","favors","headwinds","tailwinds",
    "we prefer","we like","we avoid","looks cheap","looks expensive",
    "worth considering","risk/reward","asymmetric",
]

THEME_THRESHOLD = 2.0


def _score_themes(text_lower: str) -> dict:
    """Returns {theme: score} for all themes."""
    words = text_lower.split()
    total = max(len(words), 1)
    scores = {}
    for theme, keywords in THEMES.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        scores[theme] = (count / total) * 1000
    return scores


def _detect_stance(text_lower: str) -> str:
    inst = sum(1 for m in INSTITUTIONAL_MARKERS if m in text_lower)
    inv  = sum(1 for m in INVESTOR_MARKERS if m in text_lower)
    if inv > inst:
        return "investor"
    if inst > 0:
        return "institutional"
    return "research_note"


def _classify_sentence(sentence: str) -> str:
    lower = sentence.lower()
    if any(m in lower for m in EXPLICIT_TRADE_MARKERS):
        return "explicit"
    if any(m in lower for m in IMPLICIT_TRADE_MARKERS):
        return "implicit"
    return "none"


def classify_document(text: str, filename: str = "") -> dict:
    """
    Returns classification dict:
    {
        domains: list[str],
        primary_domain: str,
        stance: str,
        trade_signal: str,         # "explicit" | "implicit" | "none"
        explicit_trades: list[str],
        implicit_trades: list[str],
    }
    """
    text_lower = text.lower()

    # 1. Themes
    theme_scores = _score_themes(text_lower)
    domains = [t for t, s in theme_scores.items() if s >= THEME_THRESHOLD and t != "Sector / Other"]
    if not domains:
        domains = ["Sector / Other"]
    primary_domain = max(theme_scores, key=theme_scores.get)
    if theme_scores[primary_domain] < THEME_THRESHOLD:
        primary_domain = "Sector / Other"

    # 2. Stance
    stance = _detect_stance(text_lower)

    # 3. Trades — classify at sentence level
    sentences = re.split(r'(?<=[.!?])\s+', text)
    explicit_trades = []
    implicit_trades = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 400:
            continue
        kind = _classify_sentence(sent)
        if kind == "explicit":
            explicit_trades.append(sent)
        elif kind == "implicit":
            implicit_trades.append(sent)

    if explicit_trades:
        trade_signal = "explicit"
    elif implicit_trades:
        trade_signal = "implicit"
    else:
        trade_signal = "none"

    return {
        "domains":        domains,
        "primary_domain": primary_domain,
        "stance":         stance,
        "trade_signal":   trade_signal,
        "explicit_trades": explicit_trades[:5],
        "implicit_trades": implicit_trades[:5],
    }
```

- [ ] **Step 2 : Tester**

```bash
python -c "
from module2_nlp.analysis.document_classifier import classify_document

# Test Goldman-style text
text = '''
We maintain Overweight on Chinese energy names with a target price of 42 USD.
Buy on weakness in WTI crude — our target is 38 dollars near-term.
Iran strike probability revised upward to 35 percent versus 20 percent prior.
Gold well-positioned as safe haven demand accelerates amid geopolitical tensions.
Fed expected to cut rates twice in H2 2026 according to our base case.
'''
result = classify_document(text)
print(f'Domains: {result[\"domains\"]}')
print(f'Primary: {result[\"primary_domain\"]}')
print(f'Stance: {result[\"stance\"]}')
print(f'Trade signal: {result[\"trade_signal\"]}')
print(f'Explicit trades: {len(result[\"explicit_trades\"])}')
assert result['trade_signal'] == 'explicit'
assert len(result['explicit_trades']) >= 1
assert 'Oil / Energy' in result['domains'] or 'Macro / Rates' in result['domains']
print('Task B OK')
"
```

- [ ] **Step 3 : Commit**

```bash
git add module2_nlp/analysis/document_classifier.py
git commit -m "feat: document classifier (domain themes + stance + trade signal detection)"
```

---

## Task C — Summarizer

**Files:**
- Create: `module2_nlp/analysis/summarizer.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/summarizer.py`**

```python
"""
Extractive document summarizer using TF-IDF sentence ranking.
No LLM — pure sklearn + regex.
Priority order: explicit trades > implicit trades > key info sentences.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from module2_nlp.analysis.document_classifier import (
    EXPLICIT_TRADE_MARKERS, IMPLICIT_TRADE_MARKERS, _classify_sentence
)

MIN_SENTENCE_LEN = 25
MAX_SENTENCE_LEN = 350


def _split_sentences(text: str) -> list:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [
        s.strip() for s in raw
        if MIN_SENTENCE_LEN <= len(s.strip()) <= MAX_SENTENCE_LEN
    ]


def _rank_by_tfidf(sentences: list) -> list:
    """Returns sentences sorted by TF-IDF importance (descending)."""
    if len(sentences) < 2:
        return sentences
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=500)
        matrix = vec.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).flatten()
        ranked = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
        return [s for _, s in ranked]
    except Exception:
        return sentences


def generate_summary(text: str, max_bullets: int = 10) -> list:
    """
    Returns list of {text: str, type: "explicit"|"implicit"|"info"} dicts.
    Ordered by: explicit trades first, then implicit, then key info.
    Max length: max_bullets.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    explicit = []
    implicit = []
    info = []

    for sent in sentences:
        kind = _classify_sentence(sent)
        if kind == "explicit":
            explicit.append({"text": sent, "type": "explicit"})
        elif kind == "implicit":
            implicit.append({"text": sent, "type": "implicit"})
        else:
            info.append(sent)

    # Rank info sentences by TF-IDF
    info_ranked = _rank_by_tfidf(info)
    info_bullets = [{"text": s, "type": "info"} for s in info_ranked]

    # Build final list respecting max_bullets
    bullets = explicit + implicit + info_bullets
    return bullets[:max_bullets]
```

- [ ] **Step 2 : Tester**

```bash
python -c "
from module2_nlp.analysis.summarizer import generate_summary

text = '''
We maintain Overweight on Chinese energy names with a target price of 42 USD.
Gold looks attractive given current geopolitical tensions in the Middle East.
Iran strike probability revised upward to 35 percent versus 20 percent prior.
Fed expected to cut rates twice in H2 2026 according to our base case model.
Hormuz closure scenario implies a 22 dollar per barrel premium on Brent crude oil.
Buy on weakness in WTI crude oil given supply disruption risk and OPEC discipline.
European gas markets face significant headwinds from potential Russian supply cuts.
China PBOC signaled moderately loose monetary policy with counter-cyclical adjustments.
Equity markets remain well-positioned for a soft landing scenario in the US.
'''

bullets = generate_summary(text, max_bullets=10)
print(f'Generated {len(bullets)} bullets')
for b in bullets:
    icon = '🟥' if b['type'] == 'explicit' else ('🟡' if b['type'] == 'implicit' else '⚪')
    print(f'  {icon} [{b[\"type\"]}] {b[\"text\"][:80]}')

assert len(bullets) <= 10
explicit_first = bullets[0]['type'] == 'explicit' if bullets else True
assert explicit_first, 'Explicit trades should be first'
print('Task C OK')
"
```

- [ ] **Step 3 : Commit**

```bash
git add module2_nlp/analysis/summarizer.py
git commit -m "feat: TF-IDF extractive summarizer with trade-priority bullet points"
```

---

## Task D — Corpus Store + Ingest Script (complet, intégrant v2)

**Files:**
- Create: `module2_nlp/analysis/corpus_store.py`
- Create: `scripts/ingest_corpus.py`

- [ ] **Step 1 : Créer `module2_nlp/analysis/corpus_store.py`**

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
    if not chunks:
        return 0
    collection = _get_collection()
    try:
        existing = collection.get(where={"doc_id": doc_id}, limit=1)
        if existing and existing["ids"]:
            return 0
    except Exception:
        pass
    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return len(chunks)


def semantic_search(query: str, n_results: int = 5) -> list:
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
    try:
        return _get_collection().count()
    except Exception:
        return 0
```

- [ ] **Step 2 : Créer `scripts/ingest_corpus.py`**

```python
#!/usr/bin/env python
"""
Ingests all PDFs from data_corpus/ into:
- SQLite via existing pipeline (NER + sentiment → entity_sentiments, document_signals)
- ChromaDB via corpus_store (semantic search)
- document_analysis table (domain/stance/trade/summary)

Idempotent: skips already-processed files via corpus_ingested table.
Usage: python scripts/ingest_corpus.py [--force]
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from module2_nlp.ingestion.pdf_parser import PdfParser
from module2_nlp.nlp.pipeline import process_document
from module2_nlp.nlp.chunker import chunk_text
from module2_nlp.analysis.corpus_store import add_document_chunks
from module2_nlp.analysis.document_classifier import classify_document
from module2_nlp.analysis.summarizer import generate_summary
from shared.db.database import (
    init_db, get_ingested_files, mark_file_ingested,
    save_document_analysis,
)

CORPUS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data_corpus')
)

_DOC_TYPE_RULES = [
    (["fomc", "federal reserve", "fed minutes"], "fomc_minutes"),
    (["earnings", "q1 ", "q2 ", "q3 ", "q4 "], "earnings_call"),
    (["howell", "campbell", "bexelius", "cascade"], "hedge_fund_letter"),
    (["goldman", "bofa", "natixis", "macquarie",
      "canaccord", "cavendish", "dbs", "seb", "bof"], "research_note"),
]


def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    for keywords, doc_type in _DOC_TYPE_RULES:
        if any(kw in name for kw in keywords):
            return doc_type
    return "news"


def _infer_source(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return base.split('-')[0].replace('_', ' ').strip()


def _infer_title(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    parts = base.split('-')
    return parts[1].replace('_', ' ').strip() if len(parts) >= 2 else base.replace('_', ' ')


def run_ingest(force: bool = False) -> dict:
    init_db()
    parser = PdfParser()

    already_ingested = set() if force else get_ingested_files()
    pdf_files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith('.pdf'))
    new_files = [f for f in pdf_files if f not in already_ingested]

    print(f"Found {len(pdf_files)} PDFs — {len(new_files)} to ingest "
          f"({'force' if force else 'incremental'})")

    results = {"processed": 0, "skipped": 0, "errors": []}

    for i, filename in enumerate(new_files):
        path = os.path.join(CORPUS_DIR, filename)
        print(f"\n  [{i+1}/{len(new_files)}] {filename}")
        try:
            doc = parser.parse(path)
            text = doc.get("text", "").strip()
            if len(text) < 100:
                print(f"    ⚠ Skipping: too short ({len(text)} chars)")
                results["skipped"] += 1
                continue

            doc_type  = _infer_doc_type(filename)
            source    = _infer_source(filename)
            title     = doc.get("title") or _infer_title(filename)
            published = doc.get("published_at")

            # 1. NLP pipeline → SQLite (entity sentiments + doc signals)
            pipeline_result = process_document(
                text=text, file_type="pdf", doc_type=doc_type,
                model="vader", title=title, source=source, published_at=published,
            )
            doc_id = pipeline_result["document"]["id"]

            # 2. ChromaDB chunks
            chunks = chunk_text(text)
            n_chunks = add_document_chunks(
                doc_id=doc_id,
                chunks=chunks,
                metadata={"source": source, "doc_type": doc_type,
                          "title": title, "filename": filename},
            )

            # 3. Document classification (domain + stance + trade)
            classification = classify_document(text, filename)

            # 4. Extractive summary
            bullets = generate_summary(text, max_bullets=10)

            # 5. Save to document_analysis
            save_document_analysis({
                "doc_id":         doc_id,
                "domains":        json.dumps(classification["domains"]),
                "primary_domain": classification["primary_domain"],
                "stance":         classification["stance"],
                "trade_signal":   classification["trade_signal"],
                "explicit_count": len(classification["explicit_trades"]),
                "implicit_count": len(classification["implicit_trades"]),
                "summary_json":   json.dumps(bullets),
            })

            # 6. Mark ingested
            mark_file_ingested(filename)
            results["processed"] += 1
            chroma_status = "added" if n_chunks > 0 else "already in ChromaDB"
            print(f"    ✅ entities={len(pipeline_result['signals'])}, "
                  f"chunks={len(chunks)} ({chroma_status}), "
                  f"domain={classification['primary_domain']}, "
                  f"trade={classification['trade_signal']}")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            results["errors"].append({"file": filename, "error": str(e)})

    print(f"\n{'='*50}")
    print(f"Done: {results['processed']} processed, {results['skipped']} skipped, "
          f"{len(results['errors'])} errors")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_ingest(force=args.force)
```

- [ ] **Step 3 : Tester corpus_store (sans ingest)**

```bash
python -c "
from module2_nlp.analysis.corpus_store import add_document_chunks, semantic_search, corpus_count
n = add_document_chunks('test001', ['Oil prices rose on OPEC news.', 'Gold surged as safe haven.'],
    {'source': 'Test', 'doc_type': 'news', 'title': 'Test', 'filename': 'test.pdf'})
assert n == 2
n2 = add_document_chunks('test001', ['dup'], {'source': 'x', 'doc_type': 'n', 'title': '', 'filename': ''})
assert n2 == 0
results = semantic_search('OPEC oil cut', n_results=2)
assert len(results) > 0 and 'similarity' in results[0]
print(f'corpus_store OK — {corpus_count()} chunks')
"
```

- [ ] **Step 4 : Lancer l'ingest complet (~3-5 min)**

```bash
python scripts/ingest_corpus.py
```

- [ ] **Step 5 : Vérifier résultats**

```bash
python -c "
from shared.db.database import get_ingested_files, get_all_entities, get_all_document_analyses
from module2_nlp.analysis.corpus_store import corpus_count
files = get_ingested_files()
entities = get_all_entities()
analyses = get_all_document_analyses()
print(f'Files ingested: {len(files)}')
print(f'Entities: {len(entities)}')
print(f'ChromaDB chunks: {corpus_count()}')
print(f'Document analyses: {len(analyses)}')
for a in analyses:
    icon = '🟥' if a['trade_signal'] == 'explicit' else ('🟡' if a['trade_signal'] == 'implicit' else '⚪')
    print(f'  {icon} {a[\"source\"]:25} | {a[\"primary_domain\"]:20} | {a[\"stance\"]}')
assert len(files) > 0
assert corpus_count() > 0
assert len(analyses) > 0
print('Task D OK')
"
```

- [ ] **Step 6 : Commit**

```bash
git add module2_nlp/analysis/corpus_store.py scripts/ingest_corpus.py
git commit -m "feat: ChromaDB corpus store + full ingest script with classification and summaries"
```

---

## Task E — Dashboard Tab 1 : Macro Digest (v2)

**Files:**
- Modify: `dashboard/app.py` (réécriture complète — commence ici)

- [ ] **Step 1 : Réécrire `dashboard/app.py` — structure + loaders + Tab 1**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import ASSETS, ALERT_THRESHOLD

st.set_page_config(
    page_title="Macro Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def get_vader():
    from shared.nlp.vader_sentiment import VaderSentiment
    return VaderSentiment()

@st.cache_resource
def get_finbert():
    from shared.nlp.finbert_sentiment import FinBERTSentiment
    return FinBERTSentiment()

def get_analyzer(name):
    return get_finbert() if name == "FinBERT" else get_vader()

@st.cache_data(ttl=3600, show_spinner="Analysing tweets…")
def load_enriched_tweets(model_name="VADER") -> pd.DataFrame:
    from module1_twitter.twitter.client import TwitterClient
    from module1_twitter.nlp.preprocessor import clean_tweet
    from module2_nlp.analysis.consensus import classify_tweet
    from module2_nlp.analysis.cross_source import THEMES as THEME_MAP

    client   = TwitterClient(backend="csv")
    tweets   = client.get_all()
    analyzer = get_analyzer(model_name)

    # Keyword → theme mapping for each tweet
    theme_kw_map = {kw.lower(): theme for theme, kws in THEME_MAP.items() for kw in kws}

    rows = []
    for t in tweets:
        clean  = clean_tweet(t["text"])
        result = analyzer.analyze(clean)
        meta   = classify_tweet(t["text"])

        # Assign theme by first matching keyword
        tweet_theme = "Other"
        for kw, theme in theme_kw_map.items():
            if kw in t["text"].lower():
                tweet_theme = theme
                break

        rows.append({
            "id":              t["id"],
            "created_at":      t["created_at"],
            "author":          t["author"],
            "text":            t["text"],
            "text_clean":      clean,
            "sentiment_score": result["score"],
            "sentiment_label": result["label"],
            "view_type":       meta["label"],
            "theme":           tweet_theme,
        })
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

@st.cache_data(ttl=3600, show_spinner="Fitting topic model…")
def load_tweet_topics(model_name="VADER"):
    from module2_nlp.analysis.topic_model import fit_tweet_topics
    df = load_enriched_tweets(model_name)
    try:
        tm, topics, labels = fit_tweet_topics(df["text"].tolist(), n_topics=8)
        return tm, topics, labels
    except Exception as e:
        return None, [-1]*len(df), {-1: "All"}

@st.cache_data(ttl=3600)
def load_corpus_analyses() -> list:
    from shared.db.database import get_all_document_analyses
    return get_all_document_analyses()

@st.cache_data(ttl=3600)
def load_corpus_entity_signals() -> pd.DataFrame:
    from shared.db.database import get_session, EntitySentiment, Document
    session = get_session()
    try:
        rows = session.query(EntitySentiment, Document).join(
            Document, EntitySentiment.document_id == Document.id).all()
        records = [{"entity": es.entity.lower(),
                    "source": (doc.source or doc.title or "?")[:30],
                    "score":  es.sentiment_score,
                    "doc_type": doc.doc_type or "unknown"}
                   for es, doc in rows]
        return pd.DataFrame(records) if records else pd.DataFrame(columns=["entity","source","score","doc_type"])
    finally:
        session.close()

def build_entity_heatmap(tweets_df, corpus_df):
    from module2_nlp.analysis.cross_source import THEMES
    all_keywords = {kw.lower() for kws in THEMES.values() for kw in kws}
    tweet_recs = []
    for kw in all_keywords:
        mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
        m = tweets_df[mask]
        if len(m) >= 3:
            tweet_recs.append({"entity": kw, "source": "📡 Tweets", "score": m["sentiment_score"].mean()})
    parts = []
    if not corpus_df.empty:
        parts.append(corpus_df[["entity","source","score"]])
    if tweet_recs:
        parts.append(pd.DataFrame(tweet_recs))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    pivot = combined.groupby(["entity","source"])["score"].mean().reset_index().pivot(
        index="entity", columns="source", values="score")
    cols = [c for c in pivot.columns if c != "📡 Tweets"] + (["📡 Tweets"] if "📡 Tweets" in pivot.columns else [])
    return pivot[cols]

# ── Sidebar global ────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧭 Macro Intelligence")
    st.divider()
    global_model = st.radio("Sentiment Model", ["VADER", "FinBERT"], key="global_model")
    from module2_nlp.analysis.cross_source import THEMES
    all_themes = list(THEMES.keys()) + ["Sector / Other"]
    sel_themes = st.multiselect("Theme filter", all_themes, default=all_themes, key="global_themes")
    trade_filter = st.selectbox("Trade signal", ["All", "With trade", "Explicit only"], key="global_trade")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🧭 Macro Digest", "📡 Tweet Intelligence", "📚 Corpus Analysis", "📊 Backtest"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MACRO DIGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🧭 Macro Digest")
    st.caption("Combined view — FinancialJuice tweets (4 days) + 14 macro documents")

    try:
        tweets_df  = load_enriched_tweets(global_model)
        corpus_df  = load_corpus_entity_signals()
        analyses   = load_corpus_analyses()
        topic_model, topics, labels = load_tweet_topics(global_model)

        # Apply theme filter to corpus
        if sel_themes and analyses:
            analyses_f = [a for a in analyses if any(d in sel_themes for d in json.loads(a.get("domains","[]")))]
        else:
            analyses_f = analyses

        # Apply trade filter
        if trade_filter == "Explicit only":
            analyses_f = [a for a in analyses_f if a["trade_signal"] == "explicit"]
        elif trade_filter == "With trade":
            analyses_f = [a for a in analyses_f if a["trade_signal"] in ("explicit","implicit")]

        # ── KPI Row ───────────────────────────────────────────────────────────
        from shared.db.database import get_ingested_files
        from module2_nlp.analysis.consensus import detect_divergences, detect_consensus
        n_docs    = len(get_ingested_files())
        n_topics  = len(set(t for t in topics if t != -1))

        # Build unified entity signals
        entity_signals = []
        if not corpus_df.empty:
            for _, r in corpus_df.iterrows():
                entity_signals.append({"entity": r["entity"], "source": r["source"], "score": r["score"]})
        from module2_nlp.analysis.cross_source import THEMES as THEME_MAP
        for kw in {k.lower() for kws in THEME_MAP.values() for k in kws}:
            mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
            m = tweets_df[mask]
            if len(m) >= 3:
                entity_signals.append({"entity": kw, "source": "📡 Tweets", "score": m["sentiment_score"].mean()})

        divergences   = detect_divergences(entity_signals, threshold=0.4)
        consensus_list = detect_consensus(entity_signals, min_sources=2)

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Tweets analysés", len(tweets_df))
        k2.metric("Docs ingérés", n_docs)
        k3.metric("Topics", n_topics)
        k4.metric("Consensus", len(consensus_list))
        k5.metric("Divergences", len(divergences))

        st.divider()

        # ── Heatmap + Digest ──────────────────────────────────────────────────
        col_heat, col_digest = st.columns([3, 2])

        with col_heat:
            st.subheader("Entity × Source Heatmap")
            heatmap_pivot = build_entity_heatmap(tweets_df, corpus_df)
            if heatmap_pivot.empty:
                st.info("Run `python scripts/ingest_corpus.py` first.")
            else:
                # Add trade badge to column labels via hover — use annotations instead
                fig_heat = px.imshow(
                    heatmap_pivot,
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    zmin=-1, zmax=1,
                    aspect="auto",
                    title="Sentiment score per entity & source (red=bearish, green=bullish)",
                )
                fig_heat.update_layout(height=520, margin=dict(t=40,b=10))
                fig_heat.update_xaxes(tickangle=45)
                st.plotly_chart(fig_heat, use_container_width=True)

        with col_digest:
            st.subheader("📋 Digest")

            # Consensus
            st.markdown("**📊 Consensus Views**")
            for c in consensus_list[:3]:
                icon = "🟢" if c["direction"] == "bullish" else "🔴"
                st.info(f"{icon} **{c['entity'].upper()}** — {c['direction']} ({c['n_sources']} sources, avg {c['mean_score']:+.2f})")
            if not consensus_list:
                st.info("No strong consensus.")

            # Divergences
            st.markdown("**⚡ Divergences**")
            for d in divergences[:3]:
                st.warning(f"**{d['entity'].upper()}** — {d['doc_a']} ({d['score_a']:+.2f}) vs {d['doc_b']} ({d['score_b']:+.2f}) · Δ={d['delta']:.2f}")
            if not divergences:
                st.warning("No major divergences.")

            # Trades from corpus ← NEW
            st.markdown("**🟥 Explicit Trades**")
            all_explicit = []
            for a in analyses_f:
                if a["trade_signal"] == "explicit" and a.get("summary_json"):
                    bullets = json.loads(a["summary_json"])
                    for b in bullets:
                        if b["type"] == "explicit":
                            all_explicit.append({"source": a["source"], "text": b["text"]})
            if all_explicit:
                for t in all_explicit[:4]:
                    st.success(f"🟥 **{t['source']}** — {t['text'][:120]}")
            else:
                st.success("No explicit trades detected.")

            st.markdown("**🟡 Implicit Trades**")
            all_implicit = []
            for a in analyses_f:
                if a["trade_signal"] in ("explicit","implicit") and a.get("summary_json"):
                    bullets = json.loads(a["summary_json"])
                    for b in bullets:
                        if b["type"] == "implicit":
                            all_implicit.append({"source": a["source"], "text": b["text"]})
            if all_implicit:
                for t in all_implicit[:3]:
                    st.warning(f"🟡 **{t['source']}** — {t['text'][:120]}")
            else:
                st.warning("No implicit trades.")

            # Weak signals
            st.markdown("**🔍 Weak Signals**")
            if not corpus_df.empty:
                corpus_ents = set(corpus_df["entity"].unique())
                tweet_kws = set()
                for kw in {k.lower() for kws in THEME_MAP.values() for k in kws}:
                    mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
                    if mask.sum() >= 3:
                        tweet_kws.add(kw)
                weak = corpus_ents - tweet_kws
                for w in list(weak)[:2]:
                    st.success(f"🔍 **{w}** — in corpus, low tweet volume")
            if corpus_df.empty:
                st.success("Ingest corpus to detect weak signals.")

        st.divider()

        # ── Cross-source alignment ─────────────────────────────────────────────
        st.subheader("Cross-Source Alignment: Tweets ↔ Corpus")
        try:
            from module2_nlp.analysis.cross_source import get_all_theme_alignments
            with st.spinner("Computing…"):
                alignments = get_all_theme_alignments(tweets_df["text"].tolist())
            if alignments:
                adf = pd.DataFrame([
                    {"Theme": k, "Alignment": v,
                     "Status": "Aligned" if v > 0.6 else ("Mixed" if v > 0.4 else "Divergent")}
                    for k, v in sorted(alignments.items(), key=lambda x: x[1], reverse=True)
                ])
                fig_a = px.bar(adf, x="Alignment", y="Theme", orientation="h",
                               color="Status",
                               color_discrete_map={"Aligned":"#22c55e","Mixed":"#f59e0b","Divergent":"#ef4444"},
                               range_x=[0,1])
                fig_a.add_vline(x=0.6, line_dash="dash", line_color="green")
                fig_a.add_vline(x=0.4, line_dash="dash", line_color="orange")
                fig_a.update_layout(height=320, margin=dict(t=30,b=10))
                st.plotly_chart(fig_a, use_container_width=True)
            else:
                st.info("Corpus needed for alignment.")
        except Exception as e:
            st.error(f"Alignment error: {e}")

    except Exception as e:
        st.error(f"Tab 1 error: {e}")
        st.exception(e)
```

- [ ] **Step 2 : Smoke test Tab 1**

```bash
python -c "
import subprocess, sys, time
p = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py', '--server.headless', 'true', '--server.port', '8502'])
time.sleep(15)
import urllib.request
try:
    urllib.request.urlopen('http://localhost:8502/_stcore/health')
    print('Tab 1 smoke test OK')
except Exception as e:
    print(f'FAILED: {e}')
finally:
    p.terminate()
"
```

- [ ] **Step 3 : Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard Tab 1 Macro Digest v2 (trades listed, domain filter)"
```

---

## Task F — Dashboard Tabs 2, 3, 4 (v2)

**Files:**
- Modify: `dashboard/app.py` — ajouter les 3 tabs restants

- [ ] **Step 1 : Ajouter Tab 2 (Tweet Intelligence v2)**

Après le bloc `with tab1:`, ajouter :

```python
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TWEET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📡 Tweet Intelligence")

    try:
        tweets_df = load_enriched_tweets(global_model)
        topic_model, topics, labels = load_tweet_topics(global_model)

        # Attach topics
        df2 = tweets_df.copy()
        df2["topic_id"]    = topics
        df2["topic_label"] = df2["topic_id"].map(labels).fillna("Other")

        # Map BERTopic labels to our 7 themes
        from module2_nlp.analysis.cross_source import THEMES as THEME_MAP
        def map_to_theme(label: str) -> str:
            label_lower = label.lower()
            for theme, kws in THEME_MAP.items():
                if any(kw.lower() in label_lower for kw in kws):
                    return theme
            return label  # keep raw if no match
        df2["topic_theme"] = df2["topic_label"].apply(map_to_theme)

        # ── Sidebar filters ───────────────────────────────────────────────────
        with st.sidebar:
            st.divider()
            st.markdown("**📡 Tweet Filters**")
            txt_search = st.text_input("🔍 Free text", key="t2_search")
            unique_themes = sorted(df2["topic_theme"].unique())
            sel_t2_themes = st.multiselect("Theme chips", unique_themes,
                                            default=unique_themes, key="t2_themes")
            sel_sent = st.multiselect("Sentiment",
                                       ["positive","negative","neutral"],
                                       default=["positive","negative","neutral"],
                                       key="t2_sent")
            sel_type = st.multiselect("Type",
                                       ["consensus","divergence","signal_faible","neutral"],
                                       default=["consensus","divergence","signal_faible","neutral"],
                                       key="t2_type")
            min_date = df2["created_at"].min().date()
            max_date = df2["created_at"].max().date()
            date_range = st.slider("Date range",
                                    min_value=min_date, max_value=max_date,
                                    value=(min_date, max_date), key="t2_dates")

        # Apply filters
        mask = (
            df2["topic_theme"].isin(sel_t2_themes) &
            df2["sentiment_label"].isin(sel_sent) &
            df2["view_type"].isin(sel_type) &
            (df2["created_at"].dt.date >= date_range[0]) &
            (df2["created_at"].dt.date <= date_range[1])
        )
        if txt_search:
            mask &= df2["text"].str.lower().str.contains(txt_search.lower(), na=False)
        filtered = df2[mask].copy()

        # ── Timeline ──────────────────────────────────────────────────────────
        st.subheader(f"Sentiment Timeline — {len(filtered):,} tweets")
        tl = (filtered.groupby([filtered["created_at"].dt.floor("H"), "sentiment_label"])
              .size().reset_index(name="count"))
        tl.columns = ["hour","sentiment","count"]
        if not tl.empty:
            fig_tl = px.line(tl, x="hour", y="count", color="sentiment",
                             color_discrete_map={"positive":"#22c55e","negative":"#ef4444","neutral":"#94a3b8"},
                             markers=True, height=200)
            fig_tl.update_layout(margin=dict(t=10,b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig_tl, use_container_width=True)

        st.divider()

        # ── Feed + Panel ──────────────────────────────────────────────────────
        col_feed, col_panel = st.columns([6, 4])

        SENT_ICON  = {"positive":"🟢","negative":"🔴","neutral":"⚪"}
        TYPE_BADGE = {"consensus":"📊","divergence":"⚡","signal_faible":"🔍","neutral":""}

        with col_feed:
            st.subheader("Feed")
            page_size = 50
            total_pages = max(1, (len(filtered)-1)//page_size + 1)
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="t2_page")
            page_df = filtered.iloc[(page-1)*page_size : page*page_size]
            st.caption(f"Showing {(page-1)*page_size+1}–{min(page*page_size, len(filtered))} of {len(filtered)}")

            for _, row in page_df.iterrows():
                icon  = SENT_ICON.get(row["sentiment_label"], "⚪")
                badge = TYPE_BADGE.get(row["view_type"], "")
                ts    = row["created_at"].strftime("%b %d %H:%M") if pd.notna(row["created_at"]) else ""
                theme_tag = f"[{row['topic_theme']}]" if row["topic_theme"] != "Other" else ""
                with st.container():
                    st.markdown(
                        f"""<div style='background:#1e1e2e;padding:10px 14px;border-radius:8px;
                        margin-bottom:8px;border-left:3px solid {"#22c55e" if row["sentiment_label"]=="positive" else "#ef4444" if row["sentiment_label"]=="negative" else "#666"}'>
                        <span style='font-size:1.1em'>{icon} {badge}</span>
                        <span style='color:#ccc;font-size:0.85em;float:right'>{ts} {theme_tag}</span><br>
                        <span style='color:#fff'>{row["text"]}</span><br>
                        <span style='color:#888;font-size:0.8em'>score: {row["sentiment_score"]:+.2f}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )

        with col_panel:
            # Donut sentiment distribution
            st.subheader("Distribution")
            sent_counts = filtered["sentiment_label"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=sent_counts.index, values=sent_counts.values,
                hole=0.5,
                marker_colors=["#22c55e" if l=="positive" else "#ef4444" if l=="negative" else "#94a3b8"
                               for l in sent_counts.index],
            ))
            fig_donut.update_layout(height=220, margin=dict(t=0,b=0,l=0,r=0),
                                     showlegend=True, legend=dict(orientation="h"))
            st.plotly_chart(fig_donut, use_container_width=True)

            # Topic chips
            st.subheader("Topics")
            THEME_COLORS = {
                "Macro / Rates":"#3b82f6","Oil / Energy":"#f59e0b",
                "Geopolitics":"#ef4444","Equities / Risk":"#22c55e",
                "China / EM":"#a855f7","Europe / FX":"#06b6d4","Other":"#6b7280"
            }
            topic_counts = filtered["topic_theme"].value_counts()
            for theme, count in topic_counts.items():
                color = THEME_COLORS.get(theme, "#6b7280")
                st.markdown(
                    f'<span style="background:{color};color:white;padding:3px 10px;'
                    f'border-radius:12px;font-size:0.85em;margin:2px;display:inline-block">'
                    f'{theme} ({count})</span>',
                    unsafe_allow_html=True
                )

            # Top entities bar
            st.subheader("Top Entities")
            import re as _re
            words = " ".join(filtered["text"]).split()
            from collections import Counter
            stops = {"the","a","an","in","on","at","to","of","for","and","or","is","are","was","were","be","been","with","from","that","this","it","its"}
            entity_words = [w.strip(".,!?:;\"'") for w in words if len(w) > 3 and w.lower() not in stops and w[0].isupper()]
            top_ents = Counter(entity_words).most_common(5)
            if top_ents:
                fig_ent = px.bar(
                    pd.DataFrame(top_ents, columns=["Entity","Count"]),
                    x="Count", y="Entity", orientation="h", height=200
                )
                fig_ent.update_layout(margin=dict(t=10,b=10))
                st.plotly_chart(fig_ent, use_container_width=True)

    except Exception as e:
        st.error(f"Tab 2 error: {e}")
        st.exception(e)
```

- [ ] **Step 2 : Ajouter Tab 3 (Corpus Analysis v2)**

```python
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORPUS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📚 Corpus Analysis")

    # Sidebar controls
    with st.sidebar:
        st.divider()
        st.markdown("**📚 Corpus Filters**")
        t3_stance = st.selectbox("Stance", ["All","institutional","investor","research_note"], key="t3_stance")
        t3_trade  = st.selectbox("Trade signal", ["All","explicit","implicit","none"], key="t3_trade")
        if st.button("🔄 Re-ingest corpus", key="t3_reingest"):
            with st.spinner("Running ingest…"):
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "scripts/ingest_corpus.py"],
                        capture_output=True, text=True,
                        cwd=os.path.join(os.path.dirname(__file__), '..')
                    )
                    if result.returncode == 0:
                        st.cache_data.clear()
                        st.sidebar.success("Done!")
                    else:
                        st.sidebar.error(result.stderr[:200])
                except Exception as e:
                    st.sidebar.error(str(e))

    try:
        corpus_df = load_corpus_entity_signals()
        analyses  = load_corpus_analyses()

        # Apply filters
        stance_f = None if t3_stance == "All" else t3_stance
        trade_f  = None if t3_trade  == "All" else t3_trade
        from shared.db.database import get_all_document_analyses
        analyses_f = get_all_document_analyses(
            domain_filter=sel_themes if sel_themes else None,
            stance_filter=stance_f,
            trade_filter=trade_f,
        )

        if corpus_df.empty and not analyses_f:
            st.warning("No corpus data. Run: `python scripts/ingest_corpus.py`")
        else:
            # ── Section 1 — Heatmap ───────────────────────────────────────────
            st.subheader("Document × Entity Sentiment Heatmap")
            if not corpus_df.empty:
                # Build source labels with badges from analyses
                trade_badges = {a["source"]: "🟥" if a["trade_signal"]=="explicit"
                                else "🟡" if a["trade_signal"]=="implicit" else "⚪"
                                for a in analyses}
                pivot_doc = (corpus_df.groupby(["source","entity"])["score"]
                             .mean().reset_index()
                             .pivot(index="source", columns="entity", values="score"))
                ent_counts = corpus_df.groupby("entity")["source"].nunique()
                keep = ent_counts[ent_counts >= 2].index
                if len(keep) > 0:
                    pivot_doc = pivot_doc[[c for c in pivot_doc.columns if c in keep]]
                # Rename index to include badge
                pivot_doc.index = [f"{trade_badges.get(s,'⚪')} {s}" for s in pivot_doc.index]
                fig_dh = px.imshow(pivot_doc, color_continuous_scale="RdYlGn",
                                   color_continuous_midpoint=0, zmin=-1, zmax=1,
                                   aspect="auto", height=450)
                fig_dh.update_layout(margin=dict(t=40,b=10))
                fig_dh.update_xaxes(tickangle=45)
                st.plotly_chart(fig_dh, use_container_width=True)

            st.divider()

            # ── Section 2 — Document Selector + Summaries ─────────────────────
            st.subheader("📄 Document Summaries")

            if analyses_f:
                col_search, col_sort = st.columns([3,1])
                with col_search:
                    doc_search = st.text_input("🔍 Search document", key="t3_doc_search")
                with col_sort:
                    doc_sort = st.selectbox("Sort by", ["Date","Alphabetical","Trade signal"], key="t3_sort")

                # Filter + sort
                disp = analyses_f
                if doc_search:
                    disp = [a for a in disp if doc_search.lower() in
                            (a.get("title","") + a.get("source","")).lower()]
                if doc_sort == "Alphabetical":
                    disp = sorted(disp, key=lambda x: (x.get("source") or ""))
                elif doc_sort == "Date":
                    disp = sorted(disp, key=lambda x: (x.get("published_at") or datetime.min), reverse=True)
                elif doc_sort == "Trade signal":
                    order = {"explicit":0,"implicit":1,"none":2}
                    disp = sorted(disp, key=lambda x: order.get(x.get("trade_signal","none"), 2))

                TRADE_ICON = {"explicit":"🟥","implicit":"🟡","none":"⚪"}
                STANCE_LABEL = {"institutional":"🏛 Institutional","investor":"💼 Investor","research_note":"📋 Research"}

                for a in disp:
                    badge   = TRADE_ICON.get(a.get("trade_signal","none"), "⚪")
                    domains = ", ".join(json.loads(a.get("domains","[]")))
                    stance  = STANCE_LABEL.get(a.get("stance",""), a.get("stance",""))
                    exp_cnt = a.get("explicit_count",0)
                    imp_cnt = a.get("implicit_count",0)
                    header  = f"{badge} **{a.get('source','?')}** — {domains} · {stance}"
                    if exp_cnt:
                        header += f" · 🟥 {exp_cnt}"
                    if imp_cnt:
                        header += f" · 🟡 {imp_cnt}"

                    with st.expander(header, expanded=False):
                        if a.get("summary_json"):
                            bullets = json.loads(a["summary_json"])
                            for b in bullets:
                                t   = b["type"]
                                ico = "🟥" if t=="explicit" else "🟡" if t=="implicit" else "⚪"
                                bg  = ("#2d1a1a" if t=="explicit"
                                       else "#2d2a1a" if t=="implicit"
                                       else "#1e1e1e")
                                st.markdown(
                                    f'<div style="background:{bg};padding:6px 10px;'
                                    f'border-radius:6px;margin:3px 0;color:#ddd">'
                                    f'{ico} {b["text"]}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No summary available — re-ingest this document.")

            st.divider()

            # ── Section 3 — Divergences ───────────────────────────────────────
            st.subheader("⚡ Inter-Document Divergences")
            if not corpus_df.empty:
                from module2_nlp.analysis.consensus import detect_divergences
                div_signals = [{"entity": r["entity"], "source": r["source"], "score": r["score"]}
                               for _, r in corpus_df.iterrows()]
                divs = detect_divergences(div_signals, threshold=0.35)
                if divs:
                    div_df = pd.DataFrame(divs)
                    div_df.columns = ["Entity","Bullish Source","Score+","Bearish Source","Score-","Δ"]
                    st.dataframe(div_df.style.background_gradient(subset=["Δ"], cmap="Reds"),
                                 use_container_width=True)
                else:
                    st.info("No significant divergences (Δ > 0.35).")

            # ── Semantic Search ───────────────────────────────────────────────
            st.subheader("🔎 Semantic Search")
            query = st.text_input("Search the corpus…", key="t3_query")
            if query:
                from module2_nlp.analysis.corpus_store import semantic_search
                with st.spinner("Searching…"):
                    results = semantic_search(query, n_results=5)
                if results:
                    for r in results:
                        with st.expander(f"📄 {r['source']} — sim: {r['similarity']:.2f}"):
                            st.markdown(r["text"])
                else:
                    st.info("No results. Ensure corpus is ingested.")

    except Exception as e:
        st.error(f"Tab 3 error: {e}")
        st.exception(e)
```

- [ ] **Step 3 : Ajouter Tab 4 (Backtest v2 avec WIP + barres sentiment)**

```python
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Backtest")
    st.warning("⚠️ **Work in Progress** — Backtest en cours de calibration sur données réelles. "
               "Les corrélations sont indicatives et non validées statistiquement.")

    # Asset keyword mapping for tweet filtering
    ASSET_KEYWORDS = {
        "WTI":     ["oil","wti","crude","opec","barrel","petroleum"],
        "Brent":   ["brent","crude","oil","opec","petroleum"],
        "Gold":    ["gold","xau","bullion","precious","safe haven"],
        "SPX":     ["s&p","spx","nasdaq","equity","stocks","recession"],
        "NASDAQ":  ["nasdaq","tech","equity","stocks"],
        "EUR/USD": ["eur","euro","ecb","europe","dollar","fx"],
    }

    bt_source = st.radio("Signal source", ["Tweets CSV", "Corpus Documents"], horizontal=True, key="bt_source")
    bt_asset  = st.selectbox("Asset", list(ASSET_KEYWORDS.keys()), key="bt_asset")

    horizons = {"15min":0.25,"30min":0.5,"1h":1,"4h":4,"12h":12,"1d":24,"3d":72,"1w":168}
    bt_hor   = st.select_slider("Horizon", options=list(horizons.keys()), value="1d", key="bt_hor")
    bt_hrs   = horizons[bt_hor]
    src_info = "📡 Alpha Vantage 1min" if bt_hrs<1 else ("📈 yfinance 1h" if bt_hrs<=24 else "📈 yfinance 1d")
    st.info(src_info)

    try:
        # Auto-detect date range from data
        tweets_df = load_enriched_tweets(global_model)
        auto_start = tweets_df["created_at"].min().date()
        auto_end   = tweets_df["created_at"].max().date()
    except Exception:
        auto_start = datetime(2024,1,1).date()
        auto_end   = datetime(2025,1,1).date()

    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("Start", value=auto_start, key="bt_start")
    with col2:
        bt_end = st.date_input("End", value=auto_end, key="bt_end")
    st.caption(f"Auto-detected from data: {auto_start} → {auto_end}")

    if bt_source == "Corpus Documents":
        from shared.db.database import get_all_entities
        entities = get_all_entities()
        bt_entity = st.selectbox("Entity", entities, key="bt_entity") if entities else None
    else:
        bt_entity = None

    if st.button("▶ Run Backtest", key="bt_run"):
        try:
            with st.spinner("Running…"):
                from shared.backtest.price_client import PriceClient
                from shared.backtest.engine import BacktestEngine

                if bt_source == "Tweets CSV":
                    # Filter tweets to asset-relevant keywords only
                    kws = ASSET_KEYWORDS.get(bt_asset, [])
                    mask = tweets_df["text"].str.lower().apply(
                        lambda t: any(k in t for k in kws))
                    asset_tweets = tweets_df[mask]
                    st.caption(f"Filtered: {len(asset_tweets)} tweets matching '{bt_asset}' out of {len(tweets_df)} total")
                    signals = [{"timestamp": r["created_at"], "signal_value": r["sentiment_score"]}
                               for _, r in asset_tweets.iterrows()]
                else:
                    if not bt_entity:
                        st.warning("Select an entity.")
                        st.stop()
                    from shared.db.database import get_doc_signals
                    raw = get_doc_signals(entity=bt_entity)
                    signals = [{"timestamp": s.get("published_at", datetime(2024,6,1)),
                                "signal_value": s["signal_value"]}
                               for s in raw if s.get("signal_value") is not None]

                if not signals:
                    st.warning("No signals found for this asset/entity.")
                else:
                    # Price chart with sentiment bars overlaid
                    try:
                        pc = PriceClient()
                        start_dt = datetime.combine(bt_start, datetime.min.time())
                        end_dt   = datetime.combine(bt_end,   datetime.max.time())
                        prices = pc.get_prices(bt_asset, start_dt, end_dt, horizon_hours=bt_hrs)

                        if not prices.empty:
                            st.subheader(f"{bt_asset} Price + Sentiment Signal")
                            fig_price = go.Figure()

                            # Price line
                            fig_price.add_trace(go.Scatter(
                                x=prices.index, y=prices["close"],
                                name="Price", line=dict(color="#60a5fa", width=2)
                            ))

                            # Sentiment bars
                            sig_df = pd.DataFrame(signals)
                            sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
                            price_min = prices["close"].min()
                            price_max = prices["close"].max()
                            price_range = max(price_max - price_min, 1)
                            bar_scale = price_range * 0.15

                            for _, row in sig_df.iterrows():
                                score = row["signal_value"]
                                color = "rgba(34,197,94,0.7)" if score > 0 else "rgba(239,68,68,0.7)"
                                height = abs(score) * bar_scale
                                fig_price.add_shape(
                                    type="rect",
                                    x0=row["timestamp"], x1=row["timestamp"] + timedelta(hours=2),
                                    y0=price_min, y1=price_min + height,
                                    fillcolor=color, line_width=0,
                                )

                            fig_price.update_layout(
                                height=400, margin=dict(t=30,b=10),
                                xaxis_title="Date", yaxis_title="Price",
                                legend=dict(orientation="h"),
                                plot_bgcolor="#0f0f1a", paper_bgcolor="#0f0f1a",
                                font=dict(color="#ccc"),
                                xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                            )
                            st.plotly_chart(fig_price, use_container_width=True)
                        else:
                            st.info("Price data not available for this period/asset.")
                    except Exception as e:
                        st.warning(f"Price chart error: {e}")

                    # KPIs + Scatter
                    engine  = BacktestEngine()
                    results = engine.run(signals, bt_asset, bt_hrs)
                    if "error" not in results and results:
                        k1,k2,k3,k4,k5 = st.columns(5)
                        k1.metric("Dir. Accuracy", f"{results['directional_accuracy']:.2%}")
                        k2.metric("Pearson r",     f"{results['pearson_r']:.3f}")
                        k3.metric("Spearman r",    f"{results['spearman_r']:.3f}")
                        k4.metric("N Signals",     results["n_signals"])
                        k5.metric("p-value",       f"{results['pearson_pval']:.4f}")

                        raw_df = results.get("raw_df")
                        if raw_df is not None and len(raw_df) > 1:
                            fig_sc = px.scatter(raw_df, x="signal", y="fwd_return",
                                                trendline="ols", height=300,
                                                title="Signal vs Forward Return")
                            st.plotly_chart(fig_sc, use_container_width=True)
                    else:
                        st.info("Not enough data for statistical analysis.")

        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.exception(e)
```

- [ ] **Step 4 : Smoke test complet**

```bash
python -c "
import subprocess, sys, time
p = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py',
    '--server.headless', 'true', '--server.port', '8503'])
time.sleep(20)
import urllib.request
try:
    urllib.request.urlopen('http://localhost:8503/_stcore/health')
    print('All tabs smoke test OK')
except Exception as e:
    print(f'FAILED: {e}')
finally:
    p.terminate()
"
```

- [ ] **Step 5 : Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard v2 — 4 tabs complete (trades, summaries, asset-filtered backtest)"
```

---

## Task G — Mise à jour CLAUDE.md + README final

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1 : Marquer toutes les tâches ✅ DONE dans CLAUDE.md**

Dans la section "🔄 SUIVI D'IMPLÉMENTATION", mettre à jour chaque ligne du tableau en ✅ DONE.

- [ ] **Step 2 : Mettre à jour README.md** — remplacer tous les ⬜ TODO par ✅ DONE

- [ ] **Step 3 : Commit final**

```bash
git add CLAUDE.md README.md
git commit -m "docs: mark all tasks complete — v2 dashboard fully implemented"
```

---

## Self-review

**Spec coverage check :**
- ✅ Domain classification (7 thèmes, TF-IDF) → Task B
- ✅ Stance detection (institutional/investor/research_note) → Task B
- ✅ Trade extraction (explicit/implicit/none) → Task B + C
- ✅ 10-bullet summaries avec hiérarchie 🟥/🟡/⚪ → Task C
- ✅ Table document_analysis → Task A
- ✅ Ingest enrichi (classifier + summarizer) → Task D
- ✅ Tab 1 : heatmap + trades listés + filtre thème → Task E
- ✅ Tab 2 : cards, compact sidebar, topics mappés → Task F
- ✅ Tab 3 : sélecteur doc search/sort + résumés + heatmap badges → Task F
- ✅ Tab 4 : WIP banner + tweets filtrés par asset + barres sentiment sur courbe → Task F
- ✅ CLAUDE.md + README → Task G

**Type consistency :**
- `classify_document()` → `{domains, primary_domain, stance, trade_signal, explicit_trades, implicit_trades}` — cohérent Tasks B/D/E/F
- `generate_summary()` → `[{text, type}]` — cohérent Tasks C/D/F
- `get_all_document_analyses()` → `[{doc_id, title, source, domains, ..., summary_json}]` — cohérent Tasks A/E/F
- `load_enriched_tweets()` → DataFrame avec colonnes `theme`, `view_type` — cohérent Tasks E/F
