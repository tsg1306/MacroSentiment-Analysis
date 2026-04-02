# Spec v2 — Macro Intelligence Dashboard
**Date :** 2026-04-02 (update post-feedback)
**Supersède :** `2026-04-02-macro-intel-redesign.md`

---

## 1. Contexte & Delta v1 → v2

Feedback reçu après première implémentation. Delta principal :

| Composant | v1 | v2 |
|-----------|----|----|
| Document classification | Aucune | Domain (7 thèmes TF-IDF) + Stance + Trade signal |
| Résumé par document | Aucun | 10 bullets extractifs, hiérarchie 🟥/🟡/⚪ |
| Tab 1 Macro Digest | Heatmap + digest | + Section trades explicites/implicites listés |
| Tab 2 Tweet Feed | Sidebar lourde, topics bruts | Sidebar compacte chips, cards propres, topics mappés |
| Tab 3 Corpus | Heatmap + divergences | + Sélecteur doc avec search/sort + résumés bullets |
| Tab 4 Backtest | Non fonctionnel | WIP banner + tweets filtrés par asset + barres sur courbe |

---

## 2. Nouveaux modules

```
module2_nlp/analysis/document_classifier.py
    → classify_document(text, filename) → {domains, primary_domain, stance, trade_signal,
                                            explicit_trades[], implicit_trades[]}

module2_nlp/analysis/summarizer.py
    → generate_summary(text, max_bullets=10) → [{text, type: explicit|implicit|info}]
```

### 2.1 document_classifier.py

**7 thèmes (TF-IDF multi-label) :**

```python
THEMES = {
    "Macro / Rates":   ["inflation","gdp","rates","central bank","monetary","fed","ecb","hike","cut","treasury"],
    "Oil / Energy":    ["oil","crude","opec","gas","barrel","refinery","hormuz","wti","brent","petroleum"],
    "Geopolitics":     ["war","sanctions","conflict","iran","israel","russia","nuclear","military","ceasefire"],
    "Equities / Risk": ["equity","stocks","s&p","earnings","recession","pe ratio","buyback","dividend","nasdaq"],
    "China / EM":      ["china","pboc","yuan","emerging markets","beijing","renminbi","gdp china"],
    "Europe / FX":     ["euro","ecb","germany","eurozone","eur","europe","european"],
    "Sector / Other":  [],  # fallback: tout ce qui ne matche aucun des 6 ci-dessus
}
```

Score par thème = nb de keywords présents / total mots × 1000. Threshold = 2.0.
Un doc peut appartenir à plusieurs thèmes. "Sector / Other" si aucun thème >= threshold.

**Stance (marqueurs lexicaux) :**
```python
INSTITUTIONAL = ["our analysts","we forecast","base case","our estimate","we expect",
                 "consensus","survey","client note","our model"]
INVESTOR      = ["i am long","we are buying","our position","we hold","added to",
                 "trimmed","initiated position","we own","personal view"]
# institutional si INSTITUTIONAL_score > INVESTOR_score, sinon investor, sinon research_note
```

**Trade signal (marqueurs lexicaux par phrase) :**
```python
EXPLICIT_TRADE = ["buy","sell","overweight","underweight","long","short",
                  "target price","price target","upgrade","downgrade","add to","trim","tp "]
IMPLICIT_TRADE = ["attractive","compelling","opportunity","well-positioned",
                  "upside","downside","favors","headwinds","tailwinds","we prefer","we like"]
```

### 2.2 summarizer.py

**Pipeline extractif (sans LLM) :**
1. Split texte en phrases (regex `(?<=[.!?])\s+`)
2. Filtrer phrases < 20 chars ou > 300 chars
3. Classifier chaque phrase : explicit | implicit | info
4. Scorer les phrases "info" par TF-IDF (sklearn TfidfVectorizer)
5. Construire les 10 bullets : explicit d'abord, puis implicit, puis info top-scored
6. Retourner `[{text: str, type: "explicit"|"implicit"|"info"}]`

---

## 3. Nouvelle table SQLite : document_analysis

```python
class DocumentAnalysis(Base):
    __tablename__ = "document_analysis"
    doc_id          = Column(Text, ForeignKey("documents.id"), primary_key=True)
    domains         = Column(Text)   # JSON list, ex: '["Macro / Rates", "Oil / Energy"]'
    primary_domain  = Column(Text)
    stance          = Column(Text)   # institutional | investor | research_note
    trade_signal    = Column(Text)   # explicit | implicit | none
    explicit_count  = Column(Integer)
    implicit_count  = Column(Integer)
    summary_json    = Column(Text)   # JSON list of {text, type}
```

Populée par `ingest_corpus.py` au moment de l'ingest de chaque PDF.

---

## 4. Dashboard — 4 tabs

### Tab 1 — 🧭 Macro Digest

**Sidebar :**
```
[ Thème ] multiselect 7 thèmes → filtre toute la page
[ Trade signal ] toggle : Tous / Avec trade (explicit+implicit) / Explicit seulement
[ Modèle ] VADER / FinBERT
```

**KPI Row (5 colonnes) :**
`Tweets analysés | Docs ingérés | Topics détectés | Consensus | Divergences`

**Row 2 — Heatmap + Digest :**

*Heatmap (gauche)* :
- Lignes = entités trackées, colonnes = sources + "📡 Tweets"
- Badge 🟥/🟡/⚪ sur chaque ligne de document selon `trade_signal`
- Filtre par thème appliqué

*Digest (droite)* :
- 📊 Consensus Views (top 3)
- ⚡ Divergences (top 3)
- 🟥 Trades explicites listés (toutes sources, avec attribution) ← NOUVEAU
- 🟡 Trades implicites listés (top 3) ← NOUVEAU
- 🔍 Signaux faibles (top 2)

**Row 3 — Cross-source alignment bar chart**

---

### Tab 2 — 📡 Tweet Intelligence

**Sidebar (compacte) :**
```
[ 🔍 Texte libre ]
[ Thème ] chips : Oil | Rates | Geo | China | Europe | Equities | Other
[ Sentiment ] toggles : 🟢 Pos  🔴 Neg  ⚪ Neu
[ Type ] toggles : 📊 📊Consensus  ⚡ Divergence  🔍 Signal
[ 📅 Date ] slider 4 jours
```

**Main area :**
1. Timeline sentiment (full width, compact, line chart h=200px)
2. Row 2 (2 colonnes 60/40) :
   - Gauche : Feed cartes (pas dividers). Chaque carte : icône + badge + date + texte + score + thème. Bouton "Charger +50"
   - Droite : donut distribution sentiment + chips topics cliquables (max 6, noms mappés vers 7 thèmes) + bar top 5 entités

**Topics mappés :** le label BERTopic brut (ex: "fed / powell / inflation") est mappé vers le thème le plus proche de nos 7. Affiché sous forme de chip coloré par thème.

---

### Tab 3 — 📚 Corpus Analysis

**Sidebar :**
```
[ Thème ] multiselect
[ Stance ] Tous / Institutionnel / Investisseur
[ Trade signal ] Tous / Explicit / Implicit / Aucun
[ Modèle ] VADER / FinBERT
[ 🔄 Re-ingest corpus ]
```

**Section 1 — Heatmap Documents × Entités**
- Badge 🟥/🟡/⚪ sur chaque doc
- Tag thème sous chaque doc name

**Section 2 — Sélecteur + Résumés** ← NOUVEAU
```
[ 🔍 Rechercher document ] [ Trier : Date ▼ | Alpha | Trade signal ]
→ Liste filtrée de documents (chips ou liste)
→ Clic sur un doc → résumé 10 bullets s'affiche dessous :

  Goldman Sachs — China Musings  [🟥 2 explicit]  [macro · energy]
  ─────────────────────────────────────────────────
  🟥 "We maintain Overweight on Chinese energy names, TP $42"
  🟥 "Buy on weakness in WTI — target $38 near-term"
  🟡 "Mideast supply creates compelling entry in oil"
  ⚪ "Iran strike probability revised to 35%"
  ...
```

**Section 3 — Divergences + Recherche sémantique** (inchangé v1)

---

### Tab 4 — 📊 Backtest

**Bannière :**
```
⚠️ Work in Progress — backtest en cours de calibration sur données réelles.
```

**Sidebar :**
```
[ Source ] Tweets CSV / Corpus Documents
[ Asset ] WTI · Brent · Gold · SPX · NASDAQ · EUR/USD
  → tweets filtrés : uniquement ceux qui mentionnent les keywords de l'asset
[ Horizon ] 15min → 1w
[ Dates ] auto-détectées depuis les données, affichées + éditables
```

**Main area :**

*Graphique principal (hero) :*
- Courbe de prix de l'asset sur la période
- Barres fines superposées : verte (tweet positif, hauteur = score), rouge (tweet négatif)
- Tooltip sur hover : texte du tweet + score

*KPIs (si ≥10 signaux) :*
`Directional Accuracy | Pearson r | N tweets filtrés / total | p-value`

*Scatter signal vs return (OLS trendline)*

---

## 5. Fichiers créés / modifiés (delta v1)

| Fichier | Action |
|---------|--------|
| `module2_nlp/analysis/document_classifier.py` | Créé |
| `module2_nlp/analysis/summarizer.py` | Créé |
| `shared/db/database.py` | Modifié — ajout `DocumentAnalysis` table |
| `scripts/ingest_corpus.py` | Modifié — appel classifier + summarizer |
| `module2_nlp/analysis/corpus_store.py` | Créé (inchangé v1) |
| `dashboard/app.py` | Réécrit complet (4 tabs v2) |

---

## 6. Contraintes

- Pas de LLM — tout NLP est local (sklearn TF-IDF, regex, sentence-transformers)
- hdbscan non disponible Python 3.14/Windows → BERTopic utilise KMeans(n_clusters=8)
- TF-IDF extractif sur phrases de 20-300 chars
- Résumé max 10 bullets, priorité : explicit > implicit > info
