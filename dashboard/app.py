"""
Macro Intelligence Dashboard v2 — Sentiment Trading Platform
4 tabs: Macro Digest | Tweet Intelligence | Corpus Analysis | Backtest
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

from config import ASSETS, ALERT_THRESHOLD

st.set_page_config(
    page_title="Macro Intelligence",
    page_icon="\U0001f9ed",
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

@st.cache_data(ttl=3600, show_spinner="Analysing tweets...")
def load_enriched_tweets(model_name="VADER") -> pd.DataFrame:
    from module1_twitter.twitter.client import TwitterClient
    from module1_twitter.nlp.preprocessor import clean_tweet
    from module2_nlp.analysis.consensus import classify_tweet
    from module2_nlp.analysis.cross_source import THEMES as THEME_MAP

    client   = TwitterClient(backend="csv")
    tweets   = client.get_all()
    analyzer = get_analyzer(model_name)

    # Keyword -> theme mapping for each tweet
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

@st.cache_data(ttl=3600, show_spinner="Fitting topic model...")
def load_tweet_topics(model_name="VADER"):
    from module2_nlp.analysis.topic_model import fit_tweet_topics
    df = load_enriched_tweets(model_name)
    try:
        tm, topics, labels = fit_tweet_topics(df["text"].tolist(), n_topics=8)
        return tm, topics, labels
    except Exception:
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

def build_entity_heatmap(tweets_df, corpus_df, max_entities=10):
    from module2_nlp.analysis.cross_source import THEMES
    all_keywords = {kw.lower() for kws in THEMES.values() for kw in kws}
    tweet_recs = []
    for kw in all_keywords:
        mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
        m = tweets_df[mask]
        if len(m) >= 3:
            tweet_recs.append({"entity": kw, "source": "Tweets", "score": m["sentiment_score"].mean()})
    parts = []
    if not corpus_df.empty:
        parts.append(corpus_df[["entity","source","score"]])
    if tweet_recs:
        parts.append(pd.DataFrame(tweet_recs))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    # Keep only top N entities by number of mentions across all sources
    entity_counts = combined.groupby("entity").size().sort_values(ascending=False)
    top_entities = entity_counts.head(max_entities).index
    combined = combined[combined["entity"].isin(top_entities)]
    pivot = combined.groupby(["entity","source"])["score"].mean().reset_index().pivot(
        index="entity", columns="source", values="score")
    cols = [c for c in pivot.columns if c != "Tweets"] + (["Tweets"] if "Tweets" in pivot.columns else [])
    return pivot[cols]

# ── Sidebar global ────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("\U0001f9ed Macro Intelligence")
    st.divider()
    global_model = st.radio("Sentiment Model", ["VADER", "FinBERT"], key="global_model")
    from module2_nlp.analysis.cross_source import THEMES
    all_themes = list(THEMES.keys()) + ["Sector / Other"]
    sel_themes = st.multiselect("Theme filter", all_themes, default=all_themes, key="global_themes")
    trade_filter = st.selectbox("Trade signal", ["All", "With trade", "Explicit only"], key="global_trade")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f9ed Macro Digest", "\U0001f4e1 Tweet Intelligence",
    "\U0001f4da Corpus Analysis", "\U0001f4ca Backtest"
])

# ==============================================================================
# TAB 1 -- MACRO DIGEST
# ==============================================================================
with tab1:
    st.header("\U0001f9ed Macro Digest")
    st.caption("Combined view -- FinancialJuice tweets (4 days) + 14 macro documents")

    try:
        tweets_df  = load_enriched_tweets(global_model)
        corpus_df  = load_corpus_entity_signals()
        analyses   = load_corpus_analyses()
        topic_model, topics, labels = load_tweet_topics(global_model)

        # Apply theme filter to corpus
        if sel_themes and analyses:
            analyses_f = [a for a in analyses if any(d in sel_themes for d in (a.get("domains") if isinstance(a.get("domains"), list) else json.loads(a.get("domains","[]"))))]
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
        from module2_nlp.analysis.cross_source import THEMES as THEME_MAP
        entity_signals = []
        if not corpus_df.empty:
            for _, r in corpus_df.iterrows():
                entity_signals.append({"entity": r["entity"], "source": r["source"], "score": r["score"]})
        for kw in {k.lower() for kws in THEME_MAP.values() for k in kws}:
            mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
            m = tweets_df[mask]
            if len(m) >= 3:
                entity_signals.append({"entity": kw, "source": "Tweets", "score": m["sentiment_score"].mean()})

        divergences    = detect_divergences(entity_signals, threshold=0.4)
        consensus_list = detect_consensus(entity_signals, min_sources=2)

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Tweets analysed", len(tweets_df))
        k2.metric("Docs ingested", n_docs)
        k3.metric("Topics", n_topics)
        k4.metric("Consensus", len(consensus_list))
        k5.metric("Divergences", len(divergences))

        st.divider()

        # ── Heatmap (full width) ─────────────────────────────────────────────
        st.subheader("Entity x Source Heatmap")
        heatmap_pivot = build_entity_heatmap(tweets_df, corpus_df)
        if heatmap_pivot.empty:
            st.info("Run `python scripts/ingest_corpus.py` first.")
        else:
            fig_heat = px.imshow(
                heatmap_pivot,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                zmin=-1, zmax=1,
                aspect="auto",
                title="Sentiment score per entity & source (red=bearish, green=bullish)",
            )
            fig_heat.update_layout(height=550, margin=dict(t=40,b=80))
            fig_heat.update_xaxes(tickangle=45)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # ── Digest — 3 columns ───────────────────────────────────────────────
        col_cons, col_trades, col_weak = st.columns(3)

        with col_cons:
            st.subheader("Consensus & Divergences")

            st.markdown("##### Consensus Views")
            if consensus_list:
                for c in consensus_list[:4]:
                    direction = "BULLISH" if c["direction"] == "bullish" else "BEARISH"
                    color = "#22c55e" if c["direction"] == "bullish" else "#ef4444"
                    st.markdown(
                        f'<div style="border-left:4px solid {color};padding:6px 12px;margin:4px 0;'
                        f'background:rgba(255,255,255,0.03);border-radius:4px">'
                        f'<b>{c["entity"].upper()}</b> — {direction}<br>'
                        f'<span style="color:#888;font-size:0.85em">'
                        f'mean={c["mean_score"]:+.2f} | {c["n_sources"]} sources | std={c["std"]:.2f}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            else:
                st.info("No strong consensus detected.")

            st.markdown("##### Divergences")
            if divergences:
                for d in divergences[:3]:
                    st.markdown(
                        f'<div style="border-left:4px solid #f59e0b;padding:6px 12px;margin:4px 0;'
                        f'background:rgba(255,255,255,0.03);border-radius:4px">'
                        f'<b>{d["entity"].upper()}</b><br>'
                        f'<span style="color:#22c55e">{d["doc_a"]} ({d["score_a"]:+.2f})</span> vs '
                        f'<span style="color:#ef4444">{d["doc_b"]} ({d["score_b"]:+.2f})</span> '
                        f'<span style="color:#888">| delta={d["delta"]:.2f}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            else:
                st.info("No major divergences.")

        with col_trades:
            st.subheader("Trade Signals")

            st.markdown("##### Explicit Trades")
            all_explicit = []
            for a in analyses_f:
                if a["trade_signal"] == "explicit" and a.get("summary_json"):
                    bullets = json.loads(a["summary_json"]) if isinstance(a["summary_json"], str) else a["summary_json"]
                    for b in bullets:
                        if b["type"] == "explicit":
                            all_explicit.append({"source": a.get("source","?"), "text": b["text"]})
            if all_explicit:
                for t in all_explicit[:4]:
                    src_short = t["source"][:20] if t["source"] else "?"
                    st.markdown(
                        f'<div style="border-left:4px solid #ef4444;padding:6px 12px;margin:4px 0;'
                        f'background:rgba(239,68,68,0.05);border-radius:4px">'
                        f'<b>{src_short}</b><br>'
                        f'<span style="font-size:0.9em">{t["text"][:150]}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            else:
                st.info("No explicit trades detected.")

            st.markdown("##### Implicit Trades")
            all_implicit = []
            for a in analyses_f:
                if a["trade_signal"] in ("explicit","implicit") and a.get("summary_json"):
                    bullets = json.loads(a["summary_json"]) if isinstance(a["summary_json"], str) else a["summary_json"]
                    for b in bullets:
                        if b["type"] == "implicit":
                            all_implicit.append({"source": a.get("source","?"), "text": b["text"]})
            if all_implicit:
                for t in all_implicit[:3]:
                    src_short = t["source"][:20] if t["source"] else "?"
                    st.markdown(
                        f'<div style="border-left:4px solid #f59e0b;padding:6px 12px;margin:4px 0;'
                        f'background:rgba(245,158,11,0.05);border-radius:4px">'
                        f'<b>{src_short}</b><br>'
                        f'<span style="font-size:0.9em">{t["text"][:150]}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            else:
                st.info("No implicit trades.")

        with col_weak:
            st.subheader("Weak Signals")
            if not corpus_df.empty:
                # Only consider tracked macro keywords (from THEMES), not all NER entities
                all_tracked_kws = {kw.lower() for kws in THEME_MAP.values() for kw in kws}
                # Which tracked keywords appear in corpus?
                corpus_tracked = set()
                corpus_scores = {}
                for kw in all_tracked_kws:
                    kw_mask = corpus_df["entity"].str.lower() == kw
                    if kw_mask.any():
                        corpus_tracked.add(kw)
                        corpus_scores[kw] = corpus_df.loc[kw_mask, "score"].mean()
                    elif corpus_df["entity"].str.lower().str.contains(kw, na=False, regex=False).any():
                        corpus_tracked.add(kw)
                        corpus_scores[kw] = corpus_df.loc[corpus_df["entity"].str.lower().str.contains(kw, na=False, regex=False), "score"].mean()
                # Which tracked keywords have low tweet coverage?
                tweet_covered = set()
                for kw in all_tracked_kws:
                    mask = tweets_df["text"].str.lower().str.contains(kw, na=False, regex=False)
                    if mask.sum() >= 5:
                        tweet_covered.add(kw)
                # Weak = in corpus tracked but low/no tweet coverage
                weak = sorted(corpus_tracked - tweet_covered, key=lambda k: abs(corpus_scores.get(k, 0)), reverse=True)
                if weak:
                    for w in weak[:5]:
                        score = corpus_scores.get(w, 0)
                        direction = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
                        color = "#22c55e" if direction == "bullish" else "#ef4444" if direction == "bearish" else "#94a3b8"
                        st.markdown(
                            f'<div style="border-left:4px solid {color};padding:6px 12px;margin:4px 0;'
                            f'background:rgba(99,102,241,0.05);border-radius:4px">'
                            f'<b>{w.upper()}</b> — {direction} in corpus ({score:+.2f}), low tweet volume</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("All tracked entities have tweet coverage.")
            else:
                st.info("Ingest corpus to detect weak signals.")

        st.divider()

        # ── Cross-source alignment ────────────────────────────────────────────
        st.subheader("Cross-Source Alignment: Tweets <-> Corpus")
        try:
            from module2_nlp.analysis.cross_source import get_all_theme_alignments
            with st.spinner("Computing..."):
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


# ==============================================================================
# TAB 2 -- TWEET INTELLIGENCE
# ==============================================================================
with tab2:
    st.header("\U0001f4e1 Tweet Intelligence")

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
            st.markdown("**Tweet Filters**")
            txt_search = st.text_input("Free text search", key="t2_search")
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
        st.subheader(f"Sentiment Timeline -- {len(filtered):,} tweets")
        tl = (filtered.groupby([filtered["created_at"].dt.floor("h"), "sentiment_label"])
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

        SENT_ICON  = {"positive":"[+]","negative":"[-]","neutral":"[=]"}
        TYPE_BADGE = {"consensus":"[C]","divergence":"[D]","signal_faible":"[W]","neutral":""}

        with col_feed:
            st.subheader("Feed")
            page_size = 50
            total_pages = max(1, (len(filtered)-1)//page_size + 1)
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="t2_page")
            page_df = filtered.iloc[(page-1)*page_size : page*page_size]
            st.caption(f"Showing {(page-1)*page_size+1}--{min(page*page_size, len(filtered))} of {len(filtered)}")

            for _, row in page_df.iterrows():
                icon  = SENT_ICON.get(row["sentiment_label"], "[=]")
                badge = TYPE_BADGE.get(row["view_type"], "")
                ts    = row["created_at"].strftime("%b %d %H:%M") if pd.notna(row["created_at"]) else ""
                theme_tag = f"[{row['topic_theme']}]" if row["topic_theme"] != "Other" else ""
                border_color = "#22c55e" if row["sentiment_label"]=="positive" else "#ef4444" if row["sentiment_label"]=="negative" else "#666"
                with st.container():
                    st.markdown(
                        f"""<div style='background:#1e1e2e;padding:10px 14px;border-radius:8px;
                        margin-bottom:8px;border-left:3px solid {border_color}'>
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
            words = " ".join(filtered["text"]).split()
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


# ==============================================================================
# TAB 3 -- CORPUS ANALYSIS
# ==============================================================================
with tab3:
    st.header("\U0001f4da Corpus Analysis")

    # Sidebar controls
    with st.sidebar:
        st.divider()
        st.markdown("**Corpus Filters**")
        t3_stance = st.selectbox("Stance", ["All","institutional","investor","research_note"], key="t3_stance")
        t3_trade  = st.selectbox("Trade signal filter", ["All","explicit","implicit","none"], key="t3_trade")
        if st.button("Re-ingest corpus", key="t3_reingest"):
            with st.spinner("Running ingest..."):
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
            # ── Section 1 -- Heatmap ─────────────────────────────────────────
            st.subheader("Document x Entity Sentiment Heatmap")
            if not corpus_df.empty:
                # Build source labels with badges from analyses
                trade_badges = {a["source"]: "[X]" if a["trade_signal"]=="explicit"
                                else "[I]" if a["trade_signal"]=="implicit" else "[ ]"
                                for a in analyses if a.get("source")}
                pivot_doc = (corpus_df.groupby(["source","entity"])["score"]
                             .mean().reset_index()
                             .pivot(index="source", columns="entity", values="score"))
                ent_counts = corpus_df.groupby("entity")["source"].nunique()
                keep = ent_counts[ent_counts >= 2].index
                if len(keep) > 0:
                    pivot_doc = pivot_doc[[c for c in pivot_doc.columns if c in keep]]
                # Rename index to include badge
                pivot_doc.index = [f"{trade_badges.get(s,'[ ]')} {s}" for s in pivot_doc.index]
                fig_dh = px.imshow(pivot_doc, color_continuous_scale="RdYlGn",
                                   color_continuous_midpoint=0, zmin=-1, zmax=1,
                                   aspect="auto", height=450)
                fig_dh.update_layout(margin=dict(t=40,b=10))
                fig_dh.update_xaxes(tickangle=45)
                st.plotly_chart(fig_dh, use_container_width=True)

            st.divider()

            # ── Section 2 -- Document Selector + Summaries ───────────────────
            st.subheader("Document Summaries")

            if analyses_f:
                col_search, col_sort = st.columns([3,1])
                with col_search:
                    doc_search = st.text_input("Search document", key="t3_doc_search")
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

                TRADE_ICON = {"explicit":"[EXPLICIT]","implicit":"[IMPLICIT]","none":"[NONE]"}
                STANCE_LABEL = {"institutional":"Institutional","investor":"Investor","research_note":"Research"}

                for a in disp:
                    badge   = TRADE_ICON.get(a.get("trade_signal","none"), "[NONE]")
                    domains_raw = a.get("domains", [])
                    if isinstance(domains_raw, str):
                        domains_raw = json.loads(domains_raw)
                    domains = ", ".join(domains_raw)
                    stance  = STANCE_LABEL.get(a.get("stance",""), a.get("stance",""))
                    exp_cnt = a.get("explicit_count",0)
                    imp_cnt = a.get("implicit_count",0)
                    header  = f"{badge} **{a.get('source','?')}** -- {domains} | {stance}"
                    if exp_cnt:
                        header += f" | Explicit: {exp_cnt}"
                    if imp_cnt:
                        header += f" | Implicit: {imp_cnt}"

                    with st.expander(header, expanded=False):
                        summary_data = a.get("summary_json")
                        if summary_data:
                            bullets = json.loads(summary_data) if isinstance(summary_data, str) else summary_data
                            for b in bullets:
                                t   = b["type"]
                                ico = "[X]" if t=="explicit" else "[I]" if t=="implicit" else "[-]"
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
                            st.info("No summary available -- re-ingest this document.")

            st.divider()

            # ── Section 3 -- Divergences ─────────────────────────────────────
            st.subheader("Inter-Document Divergences")
            if not corpus_df.empty:
                from module2_nlp.analysis.consensus import detect_divergences
                # Only keep entities mentioned by at least 2 distinct sources
                ent_src_count = corpus_df.groupby("entity")["source"].nunique()
                multi_src_entities = set(ent_src_count[ent_src_count >= 2].index)
                div_signals = [{"entity": r["entity"], "source": r["source"], "score": r["score"]}
                               for _, r in corpus_df.iterrows()
                               if r["entity"] in multi_src_entities]
                divs = detect_divergences(div_signals, threshold=0.5)
                if divs:
                    div_df = pd.DataFrame(divs[:15])  # cap at 15 rows
                    div_df.columns = ["Entity","Bullish Source","Score+","Bearish Source","Score-","Delta"]
                    st.dataframe(div_df.style.background_gradient(subset=["Delta"], cmap="Reds"),
                                 use_container_width=True)
                else:
                    st.info("No significant divergences (delta > 0.5).")

            # ── Semantic Search ───────────────────────────────────────────────
            st.subheader("Semantic Search")
            query = st.text_input("Search the corpus...", key="t3_query")
            if query:
                from module2_nlp.analysis.corpus_store import semantic_search
                with st.spinner("Searching..."):
                    results = semantic_search(query, n_results=5)
                if results:
                    for r in results:
                        with st.expander(f"{r['source']} -- sim: {r['similarity']:.2f}"):
                            st.markdown(r["text"])
                else:
                    st.info("No results. Ensure corpus is ingested.")

    except Exception as e:
        st.error(f"Tab 3 error: {e}")
        st.exception(e)


# ==============================================================================
# TAB 4 -- BACKTEST
# ==============================================================================
with tab4:
    st.header("\U0001f4ca Backtest")
    st.warning("**Work in Progress** -- Backtest en cours de calibration sur donnees reelles. "
               "Les correlations sont indicatives et non validees statistiquement.")

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
    src_info = "Alpha Vantage 1min" if bt_hrs<1 else ("yfinance 1h" if bt_hrs<=24 else "yfinance 1d")
    st.info(f"Price source: {src_info}")

    try:
        # Auto-detect date range from data
        tweets_df_bt = load_enriched_tweets(global_model)
        auto_start = tweets_df_bt["created_at"].min().date()
        auto_end   = tweets_df_bt["created_at"].max().date()
    except Exception:
        auto_start = datetime(2024,1,1).date()
        auto_end   = datetime(2025,1,1).date()

    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("Start", value=auto_start, key="bt_start")
    with col2:
        bt_end = st.date_input("End", value=auto_end, key="bt_end")
    st.caption(f"Auto-detected from data: {auto_start} -> {auto_end}")

    if bt_source == "Corpus Documents":
        from shared.db.database import get_all_entities
        entities = get_all_entities()
        bt_entity = st.selectbox("Entity", entities, key="bt_entity") if entities else None
    else:
        bt_entity = None

    if st.button("Run Backtest", key="bt_run"):
        try:
            with st.spinner("Running..."):
                from shared.backtest.price_client import PriceClient
                from shared.backtest.engine import BacktestEngine

                if bt_source == "Tweets CSV":
                    # Filter tweets to asset-relevant keywords only
                    kws = ASSET_KEYWORDS.get(bt_asset, [])
                    mask = tweets_df_bt["text"].str.lower().apply(
                        lambda t: any(k in t for k in kws))
                    asset_tweets = tweets_df_bt[mask]
                    st.caption(f"Filtered: {len(asset_tweets)} tweets matching '{bt_asset}' out of {len(tweets_df_bt)} total")
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
                            fig_sc = px.scatter(raw_df, x="signal_value", y="forward_return",
                                                trendline="ols", height=300,
                                                title="Signal vs Forward Return")
                            st.plotly_chart(fig_sc, use_container_width=True)
                    else:
                        st.info("Not enough data for statistical analysis.")

        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.exception(e)
