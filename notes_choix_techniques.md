# Notes — Choix techniques et arbitrages

## Contexte

Prototype d'analyse macro combinant un flux de tweets FinancialJuice (492 tweets exploitables sur 4 jours) et un corpus de 14 documents macro PDF (Goldman Sachs, BofA, Natixis, Macquarie, SEB, etc.).

Contrainte : zero LLM payant / API externe. Tout tourne en local.

---

## 1. Combinaison flux court + corpus long

**Approche retenue :** Espace vectoriel partage (all-MiniLM-L6-v2) pour les deux sources. Chaque source est traitee par son pipeline propre, mais les embeddings sont dans le meme espace vectoriel, ce qui permet :

- La **recherche semantique cross-source** (un tweet peut trouver le passage corpus le plus proche)
- Le **calcul d'alignement** tweets vs corpus par theme (cosine similarity entre embeddings moyens)
- La **detection de signaux faibles** : entites presentes dans le corpus mais peu/pas couvertes par les tweets

**Pourquoi pas un seul pipeline unifie ?** Les tweets et les documents ont des caracteristiques tres differentes (longueur, bruit, vocabulaire). Les tweets passent par un preprocesseur specifique (URLs, mentions, synonymes) avant l'analyse de sentiment. Les documents passent par chunking + NER + classification thematique. L'unification se fait au niveau du dashboard.

---

## 2. Sentiment : FinBERT principal, VADER en fallback

**FinBERT (ProsusAI/finbert)** : fine-tune sur 4.9M phrases financieres. Beaucoup plus precis que VADER sur le vocabulaire macro (hawkish/dovish, tightening/easing, risk-on/risk-off...).

**VADER** : enrichi avec 12 termes financiers (bullish +3, crash -3.5, surge +2.5...). Utile comme fallback instantane (pas de GPU, pas de download HuggingFace).

**Choix par defaut :** Le sidebar du dashboard permet de switcher. VADER est le defaut (plus rapide, fonctionne offline). FinBERT est recommande pour une analyse plus fine.

---

## 3. Topic modeling : BERTopic + KMeans

**Retenu :** BERTopic avec KMeans (n_clusters=8, sklearn)

**Ecarte :** LDA (mauvais sur textes courts), NMF classique, HDBSCAN (pas de wheel Windows/Python 3.14)

**Raison :** BERTopic produit des topics semantiquement coherents sur des textes courts (tweets de 1-2 phrases). Les topics emergent naturellement : Oil/Iran, ECB/Rates, Geopolitics, China/PBOC, etc.

**Note :** KMeans remplace HDBSCAN car pas de wheel precompile pour l'environnement. Impact : nombre de topics fixe (8) au lieu d'adaptatif. Extension : `pip install hdbscan` apres installation MSVC Build Tools.

---

## 4. Embeddings : all-MiniLM-L6-v2

**Retenu :** sentence-transformers all-MiniLM-L6-v2 (384 dims, ~80MB)

**Ecarte :** text-embedding-ada-002 (OpenAI, payant), TF-IDF pur (pas d'espace semantique partage)

**Raison :** Modele leger, inference en <2s sur 492 tweets CPU-only. Espace vectoriel partage tweets/corpus permettant la comparaison cross-source directe par cosine similarity.

---

## 5. Vector store : ChromaDB local

**Retenu :** ChromaDB avec persist_directory (`shared/db/chroma/`)

**Ecarte :** Faiss (pas de persistance native), Pinecone/Weaviate (cloud/payant)

**Raison :** Ingest one-shot, lecture instantanee au redemarrage. Recherche semantique sur le corpus sans LLM. Supporte les metadonnees (source, doc_type, chunk_index). ~630 chunks indexes.

---

## 6. Classification thematique des documents (7 domaines)

**Approche :** Score TF-IDF par dictionnaire de mots-cles pour chaque theme. Seuil = 2.0 (score = keyword_count / total_words * 1000). Multi-label : un document peut appartenir a plusieurs domaines.

**7 themes :**
- Macro / Rates (inflation, fed, ecb, rates, yield...)
- Oil / Energy (crude, opec, barrel, pipeline...)
- Geopolitics (war, sanctions, iran, nuclear...)
- Equities / Risk (stocks, s&p, earnings, recession...)
- China / EM (pboc, yuan, tariff, emerging...)
- Europe / FX (euro, ecb, eurozone, bund...)
- Sector / Other (fallback)

**Pourquoi pas un LLM pour classifier ?** Le TF-IDF par dictionnaire est interpretable, reproductible, et zero-cost. Sur des documents financiers avec un vocabulaire specifique, les keywords sont suffisamment discriminants. Un LLM serait plus robuste mais ajoute latence + cout + non-determinisme.

---

## 7. Detection de stance et trade signals

**Stance** (institutional / investor / research_note) : marqueurs lexicaux ("we forecast", "our estimate" -> institutional ; "i am long", "our position" -> investor). Simple et interpretable.

**Trade signals** par phrase :
- **Explicit** : "overweight", "target price", "we buy", "outperform"...
- **Implicit** : "attractive", "compelling", "upside", "we prefer"...

Les marqueurs explicites ont ete resserres pour eviter les faux positifs sur les mentions de prix historiques (ex: "sell-off" ou "long-term" ne matchent plus). La classification se fait phrase par phrase pour isoler les recommandations.

---

## 8. Resume extractif (10 bullets)

**Approche :** TF-IDF sentence ranking + tri par priorite (explicit > implicit > info).

**Pourquoi extractif et pas abstractif ?** Sur un prototype sans LLM generatif, l'extraction garantit la fidelite au texte original. Les phrases sont selectionnees par score TF-IDF (importance informationnelle) et classees par type de signal. Un trader peut lire les 10 bullets et avoir l'essentiel du document en 30 secondes.

---

## 9. Consensus / divergence detection

**3 couches complementaires :**

1. **Marqueurs lexicaux** sur tweets : dictionnaire consensus/divergence/signal faible
2. **Variance inter-sources** sur meme entite : si std > 0.3 sur les scores de sentiment de la meme entite entre les documents -> divergence
3. **Cosine similarity cross-source** : tweets vs corpus par theme (score [0-1])

**Pourquoi pas un seul mecanisme ?** Les tweets et les documents expriment le consensus/la divergence differemment. Un tweet peut explicitement dire "markets are split on..." (marqueur lexical). Deux documents peuvent avoir des scores opposes sur "oil" sans jamais le dire explicitement (variance). Le croisement tweets/corpus detecte les themes ou les deux sources sont desalignees.

---

## 10. Pipeline d'ingestion : script CLI + idempotence

**Retenu :** Script `scripts/ingest_corpus.py` + table `corpus_ingested` SQLite

**Ecarte :** Hash-check automatique au demarrage dashboard (risque de bug, complexe)

**Raison :** Fiabilite > elegance sur un prototype. Le script est idempotent (skip les fichiers deja ingeres), appelable depuis le dashboard (Tab 3 -> bouton). Ajouter un PDF = relancer le script.

---

## 11. Dashboard Streamlit, pas FastAPI + React

**Raison :** Time-to-demo 10x plus rapide. Pas besoin d'API REST pour un prototype. Streamlit permet le caching (`@st.cache_data`, `@st.cache_resource`), les widgets interactifs, et Plotly pour les visualisations, le tout en un seul fichier Python.

---

## 12. Donnees : caching et performance

- `@st.cache_data(ttl=3600)` pour les tweets et analyses corpus (rechargement si re-ingest)
- `@st.cache_resource` pour FinBERT et VADER (charges une seule fois)
- BERTopic fit sur 492 tweets en ~10s -> cache egalement

---

## Limites connues et extensions naturelles

| Limite | Extension possible |
|--------|-------------------|
| Tweets statiques (CSV snapshot) | APScheduler + polling FinancialJuice toutes les 5min |
| Pas de LLM generatif (Q&A, resume abstractif) | ChromaDB pret pour langchain + ollama (llama3.2) |
| KMeans N fixe (8 topics) | Silhouette score ou elbow method pour optimiser |
| FinBERT 512 tokens max | Chunker existant (200 mots) gere ce cas |
| 4 jours de tweets = backtest non significatif | Sur historique 3-6 mois, metriques deviennent exploitables |
| SQLite = monothread | PostgreSQL en changeant la connection string |
