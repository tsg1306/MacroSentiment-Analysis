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
