"""
Macro Intelligence Dashboard — Sentiment Trading Platform
4 tabs: Macro Digest | Tweet Intelligence | Corpus Analysis | Backtest
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

from config import (
    ASSETS, SIGNAL_WINDOWS, ALERT_THRESHOLD, TRACKED_ENTITIES, ENTITY_TICKERS,
)

st.set_page_config(page_title="Macro Intelligence Dashboard", layout="wide")
st.title("Macro Intelligence Dashboard")

# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def get_vader():
    from shared.nlp.vader_sentiment import VaderSentiment
    return VaderSentiment()

@st.cache_resource
def get_finbert():
    from shared.nlp.finbert_sentiment import FinBERTSentiment
    return FinBERTSentiment()

def get_analyzer(model_name):
    return get_finbert() if model_name == "FinBERT" else get_vader()


@st.cache_data(ttl=300)
def load_enriched_tweets(model_name):
    """Load CSV tweets, run sentiment + topic + consensus classification."""
    from module1_twitter.twitter.client import TwitterClient
    from module1_twitter.nlp.preprocessor import clean_tweet
    from module2_nlp.analysis.consensus import classify_tweet

    client = TwitterClient(backend="csv")
    raw = client.search([], limit=9999)
    analyzer = get_analyzer(model_name)

    enriched = []
    for t in raw:
        clean = clean_tweet(t["text"])
        result = analyzer.analyze(clean)
        cons = classify_tweet(clean)
        enriched.append({
            **t,
            "text_clean": clean,
            "sentiment_score": result["score"],
            "sentiment_label": result["label"],
            "confidence": result["confidence"],
            "consensus_label": cons["label"],
        })
    return enriched


@st.cache_data(ttl=300)
def run_bertopic(_texts):
    """Fit BERTopic on tweet texts. Returns (topics_list, labels_dict, model)."""
    from module2_nlp.analysis.topic_model import fit_tweet_topics
    model, topics, labels = fit_tweet_topics(list(_texts), n_topics=8)
    return topics, labels, model


@st.cache_data(ttl=600)
def load_corpus_signals():
    """Load entity signals from SQLite for all ingested documents."""
    from shared.db.database import init_db, get_session
    from shared.db.database import EntitySentiment, Document
    init_db()
    session = get_session()
    try:
        rows = session.query(
            EntitySentiment.entity,
            EntitySentiment.sentiment_score,
            EntitySentiment.document_id,
            Document.title,
            Document.source,
            Document.doc_type,
        ).join(Document, EntitySentiment.document_id == Document.id).all()
        return [
            {"entity": r[0], "score": r[1], "doc_id": r[2],
             "title": r[3], "source": r[4], "doc_type": r[5]}
            for r in rows
        ]
    finally:
        session.close()


@st.cache_data(ttl=600)
def load_doc_signals():
    """Load aggregated document signals from SQLite."""
    from shared.db.database import init_db, get_session
    from shared.db.database import DocumentSignal, Document
    init_db()
    session = get_session()
    try:
        rows = session.query(
            DocumentSignal.entity,
            DocumentSignal.signal_value,
            DocumentSignal.mention_count,
            DocumentSignal.document_id,
            DocumentSignal.published_at,
            Document.source,
            Document.doc_type,
        ).join(Document, DocumentSignal.document_id == Document.id).all()
        return [
            {"entity": r[0], "signal_value": r[1], "mention_count": r[2],
             "doc_id": r[3], "published_at": r[4], "source": r[5], "doc_type": r[6]}
            for r in rows
        ]
    finally:
        session.close()


def _short_source(source):
    """Shorten PDF filename for display."""
    if not source:
        return "Unknown"
    name = os.path.splitext(os.path.basename(source))[0]
    # Take first part before date suffix
    parts = name.split("-")
    return parts[0].replace("_", " ") if parts else name[:20]


def _color_label(val):
    if val == "positive":
        return "background-color: rgba(0,200,0,0.2)"
    elif val == "negative":
        return "background-color: rgba(200,0,0,0.2)"
    return ""


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f9ed Macro Digest",
    "\U0001f4e1 Tweet Intelligence",
    "\U0001f4da Corpus Analysis",
    "\U0001f4ca Backtest",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Macro Digest
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Macro Digest")

    model_t1 = st.radio("Sentiment model", ["VADER", "FinBERT"], key="t1_model",
                         horizontal=True)

    try:
        with st.spinner("Loading tweets & corpus data..."):
            tweets = load_enriched_tweets(model_t1)
            texts_for_bt = tuple(t["text_clean"] for t in tweets)
            topics_list, topic_labels, _bt_model = run_bertopic(texts_for_bt)

            # Assign topics to tweets
            for i, t in enumerate(tweets):
                t["topic_id"] = topics_list[i]
                t["topic_label"] = topic_labels.get(topics_list[i], f"Topic {topics_list[i]}")

            corpus_sigs = load_corpus_signals()
            doc_sigs = load_doc_signals()

        # ── KPI Row ──
        from module2_nlp.analysis.corpus_store import get_collection_count
        from shared.db.database import get_ingested_files
        from module2_nlp.analysis.consensus import detect_consensus, detect_divergences

        n_tweets = len(tweets)
        n_docs = len(get_ingested_files())
        n_topics = len(set(topics_list))
        n_chunks = get_collection_count()

        # Build entity signals for consensus/divergence from corpus
        entity_sigs_for_cd = []
        for s in doc_sigs:
            entity_sigs_for_cd.append({
                "entity": s["entity"], "source": _short_source(s["source"]),
                "score": s["signal_value"],
            })

        consensus_list = detect_consensus(entity_sigs_for_cd)
        divergence_list = detect_divergences(entity_sigs_for_cd)

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Tweets Analyzed", n_tweets)
        kpi_cols[1].metric("Docs Ingested", n_docs)
        kpi_cols[2].metric("Topics Detected", n_topics)
        kpi_cols[3].metric("Consensus", len(consensus_list))
        kpi_cols[4].metric("Divergences", len(divergence_list))

        # ── Row 2: Heatmap + Narrative ──
        col_heat, col_narr = st.columns([3, 2])

        with col_heat:
            st.markdown("**Entity x Source Heatmap**")
            # Build heatmap: rows=entities, cols=sources
            # Collect unique sources and entities
            sources_set = sorted({_short_source(s["source"]) for s in doc_sigs})
            sources_set.append("Tweets CSV")

            # Key entities to track
            key_entities = ["oil", "gold", "fed", "ecb", "iran", "opec",
                           "china", "inflation", "recession", "rates"]

            heat_data = []
            for entity in key_entities:
                row = {}
                # Doc signals
                for s in doc_sigs:
                    if s["entity"].lower() == entity:
                        src = _short_source(s["source"])
                        if src not in row:
                            row[src] = []
                        row[src].append(s["signal_value"])
                # Tweet signal (mean of matching tweets)
                matching_tweets = [t for t in tweets
                                   if entity in t["text_clean"].lower()]
                if matching_tweets:
                    tweet_mean = np.mean([t["sentiment_score"] for t in matching_tweets])
                    row["Tweets CSV"] = [tweet_mean]

                row_means = {}
                for src in sources_set:
                    vals = row.get(src, [])
                    row_means[src] = np.mean(vals) if vals else None
                heat_data.append(row_means)

            # Build matrix for plotly
            z_matrix = []
            for row_data in heat_data:
                z_matrix.append([row_data.get(src) for src in sources_set])

            fig_heat = go.Figure(go.Heatmap(
                z=z_matrix,
                x=sources_set,
                y=key_entities,
                colorscale="RdYlGn",
                zmid=0,
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z_matrix],
                texttemplate="%{text}",
                hovertemplate="Entity: %{y}<br>Source: %{x}<br>Score: %{z:.3f}<extra></extra>",
            ))
            fig_heat.update_layout(height=400, margin=dict(l=80, r=20, t=30, b=80),
                                    xaxis_tickangle=-45)
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_narr:
            st.markdown("**Digest**")

            # Consensus views
            st.markdown("##### Consensus Views")
            if consensus_list:
                for c in consensus_list[:3]:
                    direction = "BULLISH" if c["direction"] == "bullish" else "BEARISH"
                    icon = "\U0001f7e2" if c["direction"] == "bullish" else "\U0001f534"
                    st.markdown(
                        f"{icon} **{c['entity'].upper()}** — {direction} "
                        f"(mean={c['mean_score']:+.2f}, {c['n_sources']} sources, std={c['std']:.2f})"
                    )
            else:
                st.info("No strong consensus detected.")

            # Divergences
            st.markdown("##### Divergences")
            if divergence_list:
                for d in divergence_list[:3]:
                    st.markdown(
                        f"\u26a1 **{d['entity'].upper()}** — "
                        f"{d['doc_a']} ({d['score_a']:+.2f}) vs "
                        f"{d['doc_b']} ({d['score_b']:+.2f}), "
                        f"\u0394={d['delta']:.2f}"
                    )
            else:
                st.info("No divergences detected.")

            # Weak signals: entities present in corpus but not tweets, or vice versa
            st.markdown("##### Weak Signals")
            corpus_entities = {s["entity"].lower() for s in doc_sigs}
            tweet_entities_mentioned = set()
            for t in tweets:
                for ent in key_entities:
                    if ent in t["text_clean"].lower():
                        tweet_entities_mentioned.add(ent)

            only_corpus = corpus_entities.intersection(set(key_entities)) - tweet_entities_mentioned
            only_tweets = tweet_entities_mentioned - corpus_entities.intersection(set(key_entities))

            if only_corpus:
                for e in list(only_corpus)[:2]:
                    st.markdown(f"\U0001f50d **{e.upper()}** — In corpus only (not in tweets)")
            if only_tweets:
                for e in list(only_tweets)[:2]:
                    st.markdown(f"\U0001f50d **{e.upper()}** — In tweets only (not in corpus)")
            if not only_corpus and not only_tweets:
                st.info("All key entities present in both sources.")

        # ── Row 3: Cross-source alignment ──
        st.markdown("**Cross-source Alignment (Tweets vs Corpus)**")
        try:
            from module2_nlp.analysis.cross_source import get_all_theme_alignments
            with st.spinner("Computing cross-source alignment..."):
                tweet_texts_raw = [t["text_clean"] for t in tweets]
                alignments = get_all_theme_alignments(tweet_texts_raw)

            if alignments:
                themes = list(alignments.keys())
                scores = list(alignments.values())
                colors = ["green" if s > 0.6 else "orange" if s > 0.4 else "red" for s in scores]
                fig_align = go.Figure(go.Bar(
                    x=themes, y=scores,
                    marker_color=colors,
                    text=[f"{s:.2f}" for s in scores],
                    textposition="outside",
                ))
                fig_align.update_layout(
                    height=300, yaxis_range=[0, 1],
                    yaxis_title="Alignment Score",
                    margin=dict(t=20, b=40),
                )
                st.plotly_chart(fig_align, use_container_width=True)
            else:
                st.info("No alignment data available. Ingest corpus first.")
        except Exception as e:
            st.error(f"Cross-source alignment error: {e}")

    except Exception as e:
        st.error(f"Macro Digest error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tweet Intelligence
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Tweet Intelligence")

    model_t2 = st.radio("Sentiment model", ["VADER", "FinBERT"], key="t2_model",
                         horizontal=True)

    try:
        tweets_t2 = load_enriched_tweets(model_t2)
        texts_t2 = tuple(t["text_clean"] for t in tweets_t2)
        topics_t2, labels_t2, bt_model_t2 = run_bertopic(texts_t2)

        for i, t in enumerate(tweets_t2):
            t["topic_id"] = topics_t2[i]
            t["topic_label"] = labels_t2.get(topics_t2[i], f"Topic {topics_t2[i]}")

        df_tweets = pd.DataFrame(tweets_t2)

        # ── Sidebar filters ──
        with st.expander("Filters", expanded=True):
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                topic_options = sorted(labels_t2.values())
                sel_topics = st.multiselect("Topic", topic_options, default=topic_options,
                                            key="t2_topics")
            with fcol2:
                sel_sentiment = st.multiselect("Sentiment",
                                                ["positive", "negative", "neutral"],
                                                default=["positive", "negative", "neutral"],
                                                key="t2_sent")
            with fcol3:
                sel_types = st.multiselect("Type",
                                           ["consensus", "divergence", "signal_faible", "neutral"],
                                           default=["consensus", "divergence", "signal_faible", "neutral"],
                                           key="t2_types")

        # Apply filters
        mask = (
            df_tweets["topic_label"].isin(sel_topics)
            & df_tweets["sentiment_label"].isin(sel_sentiment)
            & df_tweets["consensus_label"].isin(sel_types)
        )
        df_filtered = df_tweets[mask].copy()

        # ── Timeline sentiment ──
        st.markdown("**Sentiment Timeline**")
        if not df_filtered.empty and "created_at" in df_filtered.columns:
            df_filtered["date"] = pd.to_datetime(df_filtered["created_at"])
            df_filtered = df_filtered.sort_values("date")

            # Resample by 4h bins
            df_time = df_filtered.set_index("date")
            pos_counts = df_time[df_time["sentiment_label"] == "positive"].resample("4h").size()
            neg_counts = df_time[df_time["sentiment_label"] == "negative"].resample("4h").size()
            neu_counts = df_time[df_time["sentiment_label"] == "neutral"].resample("4h").size()

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(x=pos_counts.index, y=pos_counts.values,
                                               mode="lines+markers", name="Positive",
                                               line=dict(color="green")))
            fig_timeline.add_trace(go.Scatter(x=neg_counts.index, y=neg_counts.values,
                                               mode="lines+markers", name="Negative",
                                               line=dict(color="red")))
            fig_timeline.add_trace(go.Scatter(x=neu_counts.index, y=neu_counts.values,
                                               mode="lines+markers", name="Neutral",
                                               line=dict(color="gray")))
            fig_timeline.update_layout(height=300, margin=dict(t=20, b=40),
                                        yaxis_title="Tweet count (4h bins)")
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No tweets match the current filters.")

        # ── Row 2: Feed + Topic Map ──
        col_feed, col_topic = st.columns([3, 2])

        with col_feed:
            st.markdown("**Tweet Feed**")
            if not df_filtered.empty:
                # Pagination
                page_size = 50
                total_pages = max(1, len(df_filtered) // page_size + (1 if len(df_filtered) % page_size else 0))
                page = st.number_input("Page", min_value=1, max_value=total_pages,
                                        value=1, key="t2_page")
                start_idx = (page - 1) * page_size
                page_df = df_filtered.iloc[start_idx:start_idx + page_size]

                display_df = pd.DataFrame({
                    "badge": page_df["consensus_label"].apply(
                        lambda x: "\U0001f534" if x == "divergence"
                        else "\U0001f7e2" if x == "consensus"
                        else "\U0001f7e1" if x == "signal_faible"
                        else "\u26aa"),
                    "text": page_df["text"].str[:120],
                    "score": page_df["sentiment_score"].round(3),
                    "label": page_df["sentiment_label"],
                    "topic": page_df["topic_label"],
                })
                st.dataframe(
                    display_df.style.map(_color_label, subset=["label"]),
                    use_container_width=True,
                    height=400,
                )
                st.caption(f"Showing {start_idx+1}-{min(start_idx+page_size, len(df_filtered))} "
                           f"of {len(df_filtered)} tweets")
            else:
                st.info("No tweets to display.")

        with col_topic:
            st.markdown("**Topic Map**")
            try:
                from module2_nlp.analysis.topic_model import get_topic_map
                fig_tmap = get_topic_map(bt_model_t2)
                if fig_tmap:
                    fig_tmap.update_layout(height=400, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_tmap, use_container_width=True)
                else:
                    st.info("Topic map not available.")
            except Exception:
                st.info("Topic map not available.")

        # ── Top keywords per topic ──
        st.markdown("**Top Keywords per Topic**")
        try:
            from module2_nlp.analysis.topic_model import get_topic_barchart
            fig_kw = get_topic_barchart(bt_model_t2, top_n=5)
            if fig_kw:
                fig_kw.update_layout(height=400)
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("Keyword chart not available.")
        except Exception:
            st.info("Keyword chart not available.")

    except Exception as e:
        st.error(f"Tweet Intelligence error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Corpus Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Corpus Analysis")

    # Sidebar controls
    with st.expander("Controls", expanded=True):
        ccol1, ccol2, ccol3 = st.columns(3)
        with ccol1:
            doc_model_t3 = st.radio("Model", ["VADER", "FinBERT"], key="t3_model")
        with ccol2:
            doc_type_opts = ["all", "news", "research_note", "hedge_fund_letter",
                             "earnings_call", "fomc_minutes"]
            doc_type_filter = st.selectbox("Doc type filter", doc_type_opts, key="t3_dtype")
        with ccol3:
            if st.button("Re-ingest corpus", key="t3_reingest"):
                try:
                    with st.spinner("Re-ingesting PDFs..."):
                        import subprocess
                        script = os.path.join(os.path.dirname(__file__), "..",
                                             "scripts", "ingest_corpus.py")
                        model_flag = "finbert" if doc_model_t3 == "FinBERT" else "vader"
                        result = subprocess.run(
                            [sys.executable, script, "--force", "--model", model_flag],
                            capture_output=True, text=True, timeout=600,
                        )
                        if result.returncode == 0:
                            st.success("Corpus re-ingested successfully!")
                            st.text(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                            load_corpus_signals.clear()
                            load_doc_signals.clear()
                        else:
                            st.error(f"Ingest error: {result.stderr[-300:]}")
                except Exception as e:
                    st.error(f"Re-ingest failed: {e}")

    try:
        corpus_sigs_t3 = load_corpus_signals()
        doc_sigs_t3 = load_doc_signals()

        # Apply doc_type filter
        if doc_type_filter != "all":
            corpus_sigs_t3 = [s for s in corpus_sigs_t3 if s["doc_type"] == doc_type_filter]
            doc_sigs_t3 = [s for s in doc_sigs_t3 if s["doc_type"] == doc_type_filter]

        if not doc_sigs_t3:
            st.info("No corpus data. Run ingest first: `python scripts/ingest_corpus.py`")
        else:
            # ── Heatmap Documents × Entities ──
            st.markdown("**Document x Entity Heatmap**")

            # Get unique documents and key entities
            doc_sources = sorted({_short_source(s["source"]) for s in doc_sigs_t3})
            tracked_ents = ["oil", "gold", "fed", "ecb", "iran", "opec", "china",
                            "inflation", "recession", "rates", "stocks", "crude"]

            # Build matrix
            heat_z = []
            heat_entities = []
            for ent in tracked_ents:
                row_vals = []
                has_data = False
                for src in doc_sources:
                    matches = [s["signal_value"] for s in doc_sigs_t3
                               if s["entity"].lower() == ent and _short_source(s["source"]) == src]
                    if matches:
                        row_vals.append(np.mean(matches))
                        has_data = True
                    else:
                        row_vals.append(None)
                if has_data:
                    heat_z.append(row_vals)
                    heat_entities.append(ent)

            if heat_z:
                fig_cheat = go.Figure(go.Heatmap(
                    z=heat_z, x=doc_sources, y=heat_entities,
                    colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in heat_z],
                    texttemplate="%{text}",
                ))
                fig_cheat.update_layout(height=max(250, len(heat_entities) * 35),
                                         margin=dict(l=80, r=20, t=20, b=100),
                                         xaxis_tickangle=-45)
                st.plotly_chart(fig_cheat, use_container_width=True)

            # ── Row 2: Theme bars + Emerging signals ──
            col_themes, col_emerging = st.columns(2)

            with col_themes:
                st.markdown("**Theme Scores**")
                theme_groups = {
                    "Oil / Energy": ["oil", "crude", "wti", "opec", "brent"],
                    "Rates / Macro": ["fed", "rates", "inflation", "recession", "ecb"],
                    "Geopolitics": ["iran", "russia", "sanctions", "war", "china"],
                    "Equities": ["stocks", "equities", "s&p", "spx", "nasdaq"],
                }
                theme_scores = {}
                for theme, keywords in theme_groups.items():
                    matches = [s for s in doc_sigs_t3 if s["entity"].lower() in keywords]
                    if matches:
                        weighted = sum(s["signal_value"] * s["mention_count"] for s in matches)
                        total_m = sum(s["mention_count"] for s in matches)
                        theme_scores[theme] = weighted / total_m if total_m else 0

                if theme_scores:
                    fig_themes = go.Figure(go.Bar(
                        y=list(theme_scores.keys()),
                        x=list(theme_scores.values()),
                        orientation="h",
                        marker_color=["green" if v > 0 else "red" for v in theme_scores.values()],
                        text=[f"{v:+.3f}" for v in theme_scores.values()],
                        textposition="outside",
                    ))
                    fig_themes.update_layout(height=250, margin=dict(l=100, r=40, t=10, b=10),
                                              xaxis_range=[-1, 1])
                    st.plotly_chart(fig_themes, use_container_width=True)

            with col_emerging:
                st.markdown("**Emerging Signals (top entities by mentions)**")
                # Entities outside the tracked taxonomy
                tracked_flat = set()
                for cat_ents in TRACKED_ENTITIES.values():
                    tracked_flat.update(e.lower() for e in cat_ents)

                all_ents = Counter()
                for s in doc_sigs_t3:
                    all_ents[s["entity"].lower()] += s["mention_count"]

                emerging = [(e, c) for e, c in all_ents.most_common(20)
                            if e not in tracked_flat and len(e) > 2][:10]
                if emerging:
                    fig_em = go.Figure(go.Bar(
                        y=[e[0] for e in emerging],
                        x=[e[1] for e in emerging],
                        orientation="h",
                    ))
                    fig_em.update_layout(height=250, margin=dict(l=100, r=20, t=10, b=10))
                    st.plotly_chart(fig_em, use_container_width=True)
                else:
                    st.info("No emerging signals outside tracked taxonomy.")

            # ── Divergences table ──
            st.markdown("**Inter-document Divergences**")
            from module2_nlp.analysis.consensus import detect_divergences
            entity_sigs_cd = [
                {"entity": s["entity"], "source": _short_source(s["source"]),
                 "score": s["signal_value"]}
                for s in doc_sigs_t3
            ]
            divs = detect_divergences(entity_sigs_cd, threshold=0.4)
            if divs:
                div_df = pd.DataFrame(divs[:15])
                st.dataframe(div_df, use_container_width=True)
            else:
                st.info("No divergences found (threshold 0.4).")

            # ── Semantic search ──
            st.markdown("**Semantic Search**")
            search_query = st.text_input("Search corpus passages", key="t3_search",
                                          placeholder="e.g. OPEC production cuts impact on oil prices")
            if search_query:
                try:
                    from module2_nlp.analysis.corpus_store import semantic_search
                    with st.spinner("Searching..."):
                        results = semantic_search(search_query, n_results=5)
                    if results:
                        for r in results:
                            with st.container():
                                st.markdown(
                                    f"**[{r['similarity']:.3f}]** `{r['source']}` "
                                    f"({r['doc_type']})"
                                )
                                st.caption(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
                                st.divider()
                    else:
                        st.info("No matching passages found.")
                except Exception as e:
                    st.error(f"Search error: {e}")

    except Exception as e:
        st.error(f"Corpus Analysis error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Backtest
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Backtest")

    # Controls
    with st.expander("Controls", expanded=True):
        bcol1, bcol2, bcol3 = st.columns(3)

        with bcol1:
            bt_source = st.radio("Signal source", ["Tweets CSV", "Corpus Documents"],
                                  key="bt_source")
            bt_asset = st.selectbox("Price Asset",
                                     ["WTI", "BRENT", "GOLD", "SPX", "NASDAQ", "EUR/USD"],
                                     key="bt_asset_t4")
        with bcol2:
            horizons_map = {"15min": 0.25, "30min": 0.5, "1h": 1, "4h": 4,
                            "12h": 12, "1d": 24, "3d": 72, "1w": 168}
            bt_horizon = st.select_slider("Horizon", options=list(horizons_map.keys()),
                                           value="1d", key="bt_hz")
            bt_hours = horizons_map[bt_horizon]

            if bt_hours < 1:
                st.info("Source: Alpha Vantage 1min")
            elif bt_hours <= 24:
                st.info("Source: yfinance 1h")
            else:
                st.info("Source: yfinance 1d")

        with bcol3:
            bt_start = st.date_input("Start", value=datetime(2024, 1, 1), key="bt_start_t4")
            bt_end = st.date_input("End", value=datetime(2024, 12, 31), key="bt_end_t4")

            if bt_source == "Corpus Documents":
                try:
                    from shared.db.database import get_all_entities
                    all_ents = get_all_entities()
                    bt_entity = st.selectbox("Entity", all_ents[:50] if all_ents else ["oil"],
                                              key="bt_ent")
                except Exception:
                    bt_entity = st.text_input("Entity", value="oil", key="bt_ent_txt")

    if st.button("Run Backtest", key="bt_run_t4"):
        try:
            with st.spinner("Running backtest..."):
                signals = []

                if bt_source == "Tweets CSV":
                    # Use tweet sentiment as signals
                    model_bt = st.session_state.get("t2_model", "VADER")
                    tweets_bt = load_enriched_tweets(model_bt)
                    for t in tweets_bt:
                        signals.append({
                            "timestamp": t["created_at"],
                            "signal_value": t["sentiment_score"],
                        })
                else:
                    # Corpus document signals
                    from shared.db.database import get_doc_signals
                    doc_sigs_bt = get_doc_signals(entity=bt_entity)
                    for s in doc_sigs_bt:
                        ts = s.get("published_at")
                        if ts:
                            signals.append({
                                "timestamp": ts,
                                "signal_value": s["signal_value"],
                            })

                if not signals:
                    st.warning("No signals found. Analyze data first.")
                elif len(signals) < 5:
                    st.warning(f"Only {len(signals)} signals — need at least 5 for backtest.")
                else:
                    from shared.backtest.engine import BacktestEngine
                    engine = BacktestEngine()
                    results = engine.run(signals, bt_asset, bt_hours)

                    if "error" in results:
                        st.warning(results["error"])
                    else:
                        # KPIs
                        kpi_cols = st.columns(5)
                        kpi_cols[0].metric("Directional Accuracy",
                                           f"{results['directional_accuracy']:.2%}")
                        kpi_cols[1].metric("Pearson r", f"{results['pearson_r']:.3f}")
                        kpi_cols[2].metric("Spearman r", f"{results['spearman_r']:.3f}")
                        kpi_cols[3].metric("N Signals", results["n_signals"])
                        kpi_cols[4].metric("p-value", f"{results['pearson_pval']:.4f}")

                        # Forward returns by bucket
                        buckets = results["forward_returns_by_bucket"]
                        fig_b = go.Figure(go.Bar(
                            x=list(buckets.keys()), y=list(buckets.values()),
                            marker_color=["red" if v < 0 else "green"
                                          for v in buckets.values()],
                            text=[f"{v:+.4f}" for v in buckets.values()],
                            textposition="outside",
                        ))
                        fig_b.update_layout(title="Forward Returns by Sentiment Bucket",
                                            height=350)
                        st.plotly_chart(fig_b, use_container_width=True)

                        # Row 2: Rolling correlation + Scatter
                        bt_c1, bt_c2 = st.columns(2)

                        with bt_c1:
                            rc = results["rolling_correlation"]
                            if rc:
                                rc_df = pd.DataFrame(rc)
                                fig_rc = go.Figure()
                                fig_rc.add_trace(go.Scatter(
                                    x=rc_df["timestamp"], y=rc_df["rolling_corr"],
                                    mode="lines", name="Rolling Corr",
                                ))
                                fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
                                fig_rc.update_layout(title="Rolling Correlation (20 obs)",
                                                      height=350)
                                st.plotly_chart(fig_rc, use_container_width=True)

                        with bt_c2:
                            raw_df = results.get("raw_df")
                            if raw_df is not None and len(raw_df) > 1:
                                fig_s = go.Figure()
                                fig_s.add_trace(go.Scatter(
                                    x=raw_df["signal_value"],
                                    y=raw_df["forward_return"],
                                    mode="markers", name="Signals",
                                ))
                                z = np.polyfit(raw_df["signal_value"],
                                              raw_df["forward_return"], 1)
                                x_l = np.linspace(raw_df["signal_value"].min(),
                                                  raw_df["signal_value"].max(), 50)
                                fig_s.add_trace(go.Scatter(
                                    x=x_l, y=np.polyval(z, x_l),
                                    mode="lines", name="OLS Trend",
                                    line=dict(dash="dash", color="red"),
                                ))
                                fig_s.update_layout(
                                    title="Signal vs Forward Return",
                                    height=350,
                                    xaxis_title="Signal",
                                    yaxis_title="Forward Return",
                                )
                                st.plotly_chart(fig_s, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest error: {e}")
