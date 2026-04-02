# État de l'art — Prototype d'analyse macro sans LLM payant

## Contexte du projet

Construire un prototype permettant d'exploiter deux sources hétérogènes :
- Un **flux de tweets Financial Juice** des 4 derniers jours (données courtes, haute fréquence)
- Un **corpus de 10 documents macroéconomiques** (textes longs, structurés)

Objectif : produire une lecture utile de l'actualité macro — feed interactif, thèmes émergents, consensus vs divergences, signaux faibles.

**Contrainte clé** : zéro LLM payant ou limité. Tout doit tourner en local ou via des options gratuites et illimitées.

---

## Stack technique retenue

| Couche | Outil | Raison |
|--------|-------|--------|
| Sentiment tweets | `FinBERT` (HuggingFace) | Spécialisé finance, local, gratuit |
| Embeddings sémantiques | `sentence-transformers` | Léger, offline, très performant |
| Topic modeling | `BERTopic` | Thèmes cohérents sans définir N à l'avance |
| Consensus/divergence | Cosine similarity + lexical markers | Rapide, explicable, zéro coût |
| RAG corpus | `LangChain` + `ChromaDB` | Standard de facto, in-memory possible |
| LLM optionnel | `Ollama` (llama3.2 / mistral) | Local, illimité, zéro API |
| Dashboard | `Streamlit` | Python natif, time-to-demo 10x plus court |

---

## Composant 1 — Analyse du flux tweets (NLP local)

### 1.1 Sentiment financier avec FinBERT

FinBERT est un modèle BERT fine-tuné sur 4,9 millions de phrases financières. Il classe chaque texte en `positive`, `negative`, `neutral` avec un score de confiance.

**Installation :**
```bash
pip install transformers torch
```

**Usage :**
```python
from transformers import pipeline

nlp = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

tweets = [
    "Fed signals hawkish stance, rate cuts unlikely in Q1",
    "Strong NFP data confirms labor market resilience",
    "Recession fears mount as yield curve inverts further"
]

results = nlp(tweets)
# [{'label': 'negative', 'score': 0.91}, ...]
```

**Alternative plus légère pour tweets :**
```bash
pip install tweetnlp
```
```python
import tweetnlp
model = tweetnlp.load_model('sentiment')
model.sentiment("Markets pricing in soft landing scenario")
```

### 1.2 Embeddings sémantiques avec sentence-transformers

Les embeddings permettent de mesurer la similarité sémantique entre textes — base de tout le reste (clustering, consensus, recherche).

**Installation :**
```bash
pip install sentence-transformers
```

**Usage :**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, très rapide

texts = [...]  # tweets ou passages de documents
embeddings = model.encode(texts, show_progress_bar=True)
# Shape: (n_texts, 384)
```

**Modèles recommandés :**
- `all-MiniLM-L6-v2` — léger, rapide, bon généraliste (recommandé pour démo)
- `all-mpnet-base-v2` — plus précis, plus lent
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — si besoin multilingue

### 1.3 Topic modeling avec BERTopic

BERTopic combine embeddings BERT + clustering HDBSCAN + représentation c-TF-IDF. Avantage majeur : on n'a pas besoin de définir le nombre de topics à l'avance.

**Installation :**
```bash
pip install bertopic
```

**Pipeline complet :**
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Utiliser le même modèle d'embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic_model = BERTopic(embedding_model=embedding_model, language="english")

# Fit sur les tweets
topics, probs = topic_model.fit_transform(tweets)

# Inspecter les topics
topic_model.get_topic_info()
#    Topic  Count  Name
#    0      45     Fed_rates_inflation_hike
#    1      32     China_growth_GDP_slowdown
#    2      28     Energy_oil_OPEC_supply
#    -1     12     (outliers)

# Topics les plus représentatifs
topic_model.get_topic(0)
# [('rates', 0.091), ('fed', 0.087), ('inflation', 0.082), ...]
```

**Visualisation interactive (Plotly) :**
```python
topic_model.visualize_topics()         # carte 2D des topics
topic_model.visualize_barchart()       # mots-clés par topic
topic_model.visualize_topics_over_time(tweets_df, timestamps)  # évolution temporelle
```

---

## Composant 2 — Détection consensus vs divergences (sans LLM)

C'est le cœur analytique du projet. Trois approches complémentaires, toutes gratuites.

### 2.1 Similarité cosinus (approche principale)

**Principe :** si tous les textes sur un sujet sont proches dans l'espace vectoriel → consensus. S'ils se regroupent en sous-clusters distincts → divergence.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def detect_consensus_divergence(embeddings, threshold_consensus=0.75, threshold_divergence=0.40):
    """
    Retourne un score et une catégorie pour un groupe de textes.
    
    Returns:
        dict avec 'type' (consensus/divergence/mixed), 'avg_similarity', 'clusters'
    """
    sim_matrix = cosine_similarity(embeddings)
    
    # Similarité moyenne (hors diagonale)
    n = len(embeddings)
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    
    if avg_sim >= threshold_consensus:
        return {"type": "consensus", "avg_similarity": avg_sim}
    elif avg_sim <= threshold_divergence:
        return {"type": "divergence", "avg_similarity": avg_sim}
    else:
        return {"type": "mixed", "avg_similarity": avg_sim}


# Usage par topic
for topic_id in topic_model.get_topics():
    topic_tweets = [tweets[i] for i, t in enumerate(topics) if t == topic_id]
    topic_embeddings = model.encode(topic_tweets)
    result = detect_consensus_divergence(topic_embeddings)
    print(f"Topic {topic_id}: {result['type']} (sim={result['avg_similarity']:.2f})")
```

### 2.2 Marqueurs lexicaux (complément rapide)

Une approche par dictionnaire de marqueurs linguistiques, très explicable et zéro GPU.

```python
CONSENSUS_MARKERS = [
    "confirms", "as expected", "in line with", "consensus",
    "widely expected", "markets expect", "broadly anticipated",
    "reaffirms", "consistent with"
]

DIVERGENCE_MARKERS = [
    "contrary to", "surprisingly", "unexpectedly", "despite",
    "however", "risks to the upside", "risks to the downside",
    "disagrees", "diverges", "at odds with", "questions",
    "challenges the view", "pushback"
]

SIGNAL_FAIBLE_MARKERS = [
    "first time since", "unusual", "historic", "unprecedented",
    "quietly", "under the radar", "few noticed", "overlooked"
]

def classify_tweet(text: str) -> dict:
    text_lower = text.lower()
    
    c_score = sum(1 for w in CONSENSUS_MARKERS if w in text_lower)
    d_score = sum(1 for w in DIVERGENCE_MARKERS if w in text_lower)
    s_score = sum(1 for w in SIGNAL_FAIBLE_MARKERS if w in text_lower)
    
    label = "neutral"
    if s_score > 0:
        label = "signal_faible"
    elif c_score > d_score:
        label = "consensus"
    elif d_score > c_score:
        label = "divergence"
    
    return {
        "label": label,
        "consensus_score": c_score,
        "divergence_score": d_score,
        "signal_score": s_score
    }
```

### 2.3 Sentiment contradictoire sur une même entité (approche avancée)

Détecter quand deux textes parlent de la **même entité** (Fed, BCE, inflation...) mais avec des sentiments opposés.

```python
import spacy
nlp_ner = spacy.load("en_core_web_sm")  # pip install spacy && python -m spacy download en_core_web_sm

def extract_entity_sentiment(tweets_df):
    """
    Pour chaque entité nommée, agréger les sentiments.
    Si variance haute → divergence de vues.
    """
    records = []
    for _, row in tweets_df.iterrows():
        doc = nlp_ner(row['text'])
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON']]
        for ent in entities:
            records.append({
                'entity': ent,
                'sentiment_score': row['sentiment_score'],  # issu de FinBERT
                'tweet': row['text']
            })
    
    df = pd.DataFrame(records)
    
    # Variance de sentiment par entité
    entity_stats = df.groupby('entity')['sentiment_score'].agg(['mean', 'std', 'count'])
    entity_stats['divergence'] = entity_stats['std'] > 0.3  # seuil configurable
    
    return entity_stats.sort_values('count', ascending=False)
```

---

## Composant 3 — RAG sur corpus documentaire

### 3.1 Pipeline LangChain + ChromaDB (in-memory)

```bash
pip install langchain langchain-community chromadb pypdf sentence-transformers
```

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Chargement des 10 documents
loader = DirectoryLoader('./data/corpus/', glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 2. Découpage en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)

# 3. Embeddings locaux (même modèle que pour les tweets → comparaison possible)
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# 4. Vectorstore in-memory (ou persist_directory pour persistance)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### 3.2 Q&A avec Ollama (LLM local gratuit)

Ollama permet de faire tourner des LLMs en local sans limite et sans coût.

**Installation Ollama :**
```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2   # ~2GB
# ou
ollama pull mistral    # alternative légère
```

**Intégration LangChain :**
```python
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

llm = Ollama(model="llama3.2")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Q&A sur le corpus
result = qa_chain.invoke({"query": "What are the main inflation risks discussed?"})
print(result['result'])
print([doc.metadata['source'] for doc in result['source_documents']])
```

### 3.3 Synthèse cross-sources (tweets + corpus)

Comparer ce que disent les tweets vs le corpus sur les mêmes entités/sujets :

```python
def cross_source_analysis(topic_keywords: list[str]) -> dict:
    """
    Pour un ensemble de mots-clés (ex: ['inflation', 'CPI', 'prices']),
    récupère les passages pertinents du corpus ET les tweets correspondants,
    puis calcule la divergence sémantique entre les deux sources.
    """
    query = " ".join(topic_keywords)
    
    # Passages du corpus
    corpus_docs = retriever.invoke(query)
    corpus_texts = [d.page_content for d in corpus_docs]
    
    # Tweets filtrés
    tweet_texts = [t for t in tweets if any(kw.lower() in t.lower() for kw in topic_keywords)]
    
    # Embeddings comparatifs
    corpus_emb = model.encode(corpus_texts)
    tweet_emb = model.encode(tweet_texts)
    
    # Similarité cross-source : corpus vs tweets
    cross_sim = cosine_similarity(corpus_emb, tweet_emb)
    avg_cross_sim = cross_sim.mean()
    
    return {
        "topic": query,
        "n_corpus_passages": len(corpus_texts),
        "n_tweets": len(tweet_texts),
        "cross_source_similarity": float(avg_cross_sim),
        "interpretation": "aligned" if avg_cross_sim > 0.6 else "divergent"
    }
```

---

## Composant 4 — Dashboard Streamlit

### Structure suggérée

```python
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Macro Intel Dashboard", layout="wide")
st.title("Macro Intelligence Feed")

# ── Sidebar : filtres ──
with st.sidebar:
    st.header("Filtres")
    date_range = st.date_input("Période", [])
    selected_topics = st.multiselect("Topics", options=topic_labels)

# ── Row 1 : KPIs ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tweets analysés", len(tweets_df))
col2.metric("Topics détectés", n_topics)
col3.metric("Signaux consensus", n_consensus)
col4.metric("Divergences", n_divergences)

# ── Row 2 : Feed + Topic map ──
col_feed, col_map = st.columns([1, 1])

with col_feed:
    st.subheader("Feed interactif")
    for _, tweet in filtered_tweets.iterrows():
        color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(tweet['sentiment'])
        badge = {"consensus": "📊", "divergence": "⚡", "signal_faible": "🔍"}.get(tweet['view_type'], "")
        st.markdown(f"{color} {badge} {tweet['text']}")
        st.caption(f"Score: {tweet['sentiment_score']:.2f} · Topic: {tweet['topic_label']}")
        st.divider()

with col_map:
    st.subheader("Carte des topics")
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3 : Timeline sentiment ──
st.subheader("Évolution du sentiment")
fig_timeline = px.line(
    tweets_df.groupby(['date', 'sentiment']).size().reset_index(name='count'),
    x='date', y='count', color='sentiment',
    color_discrete_map={'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#94a3b8'}
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Row 4 : Consensus / Divergences ──
col_cons, col_div = st.columns(2)
with col_cons:
    st.subheader("📊 Consensus Views")
    for item in consensus_list:
        st.info(f"**{item['topic']}** — {item['summary']}")

with col_div:
    st.subheader("⚡ Divergences")
    for item in divergence_list:
        st.warning(f"**{item['topic']}** — {item['summary']}")

# ── Row 5 : Q&A corpus ──
st.subheader("Q&A Corpus documentaire")
query = st.text_input("Posez une question sur les 10 documents")
if query:
    with st.spinner("Recherche..."):
        result = qa_chain.invoke({"query": query})
        st.write(result['result'])
        with st.expander("Sources"):
            for doc in result['source_documents']:
                st.caption(f"📄 {doc.metadata['source']} — page {doc.metadata.get('page', '?')}")
```

---

## Structure du repository

```
macro-intel/
├── data/
│   ├── tweets/
│   │   └── financial_juice_4d.csv   # flux tweets
│   └── corpus/
│       └── *.pdf                    # 10 documents macro
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py          # chargement et nettoyage des données
│   ├── sentiment.py       # FinBERT scoring
│   ├── topics.py          # BERTopic topic modeling
│   ├── consensus.py       # détection consensus/divergence
│   └── rag.py             # LangChain RAG sur corpus
│
├── analysis/
│   └── cross_source.py    # comparaison tweets vs corpus
│
├── app.py                 # dashboard Streamlit
├── run_pipeline.py        # script CLI pour lancer tout le pipeline
├── requirements.txt
└── README.md
```

---

## Requirements

```txt
# requirements.txt
pandas>=2.0
numpy>=1.24

# NLP & embeddings
transformers>=4.35
sentence-transformers>=2.2
torch>=2.0
bertopic>=0.16
spacy>=3.7

# RAG
langchain>=0.1
langchain-community>=0.0.20
chromadb>=0.4
pypdf>=3.0

# Sentiment tweets
tweetnlp>=1.0

# Visualisation
streamlit>=1.30
plotly>=5.17
```

**Installation complète :**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Lancement :**
```bash
# 1. Traitement pipeline (une fois)
python run_pipeline.py

# 2. Dashboard interactif
streamlit run app.py
```

---

## Arbitrages et justifications

| Choix | Alternative écartée | Raison |
|-------|---------------------|--------|
| FinBERT local | OpenAI API sentiment | Gratuit, illimité, spécialisé finance |
| `all-MiniLM-L6-v2` | `text-embedding-ada-002` | 80MB local vs API payante |
| BERTopic | LDA / NMF | Topics cohérents, pas besoin de définir N, visualisations natives |
| ChromaDB in-memory | Pinecone, Weaviate | Zéro infra, parfait pour prototype 6h |
| Ollama (optionnel) | GPT-4, Claude API | Local, illimité, zéro coût |
| Streamlit | FastAPI + React | Time-to-demo 10x plus court, Python natif |
| Cosine similarity | Cross-encoder | Léger, interprétable, pas de GPU requis |

---

## Limites connues

- **ChromaDB in-memory** : données perdues au redémarrage. Pour persistance : `persist_directory="./chroma_db"`
- **FinBERT** : longueur max 512 tokens. Pour les tweets c'est sans problème, pour les passages longs → chunker avant
- **BERTopic** : nécessite un minimum ~20 documents pour produire des topics stables
- **Ollama** : performances dépendantes du hardware local (CPU vs GPU)
- **Pas de streaming** des tweets : le prototype travaille sur un snapshot CSV statique

---

*Document généré pour usage avec Claude Code — toutes les dépendances sont open source et gratuites.*
