import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import ASSETS, SIGNAL_WINDOWS, ALERT_THRESHOLD, HIGH_AUTHORITY_ACCOUNTS

st.set_page_config(page_title="Sentiment Trading Platform", layout="wide")
st.title("Sentiment Trading Platform")

# ── Cached resources ──────────────────────────────────────────────────────────

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

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f4e1 Twitter Live", "\U0001f4ca Twitter Backtest",
    "\U0001f4c4 Document Analyzer", "\U0001f52c Document Backtest"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Twitter Live Signal
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    with st.sidebar:
        st.header("Twitter Settings")
        model_choice = st.radio("Model", ["VADER", "FinBERT"], key="t1_model")
        backend_choice = st.radio("Backend", ["mock", "snscrape", "api"], key="t1_backend")
        asset_choice = st.selectbox("Asset", list(ASSETS.keys()), key="t1_asset")
        default_kw = ASSETS[asset_choice]["keywords"][:5]
        keywords = st.multiselect("Keywords", ASSETS[asset_choice]["keywords"],
                                  default=default_kw, key="t1_kw")
        custom_account = st.text_input("Custom account", key="t1_account")
        refresh = st.button("\U0001f504 Refresh", key="t1_refresh")

    if refresh:
        try:
            with st.spinner("Fetching tweets..."):
                from module1_twitter.twitter.client import TwitterClient
                from module1_twitter.nlp.preprocessor import clean_tweet
                from module1_twitter.signal.extractor import compute_all_windows
                from shared.db.database import init_db, save_tweet, save_tweet_signal

                init_db()
                client = TwitterClient(backend=backend_choice)
                analyzer = get_analyzer(model_choice)

                tweets = client.search(keywords, limit=200)
                if custom_account:
                    tweets += client.get_user_tweets(custom_account, limit=20)

                enriched = []
                for t in tweets:
                    clean = clean_tweet(t["text"])
                    result = analyzer.analyze(clean)
                    t_enriched = {
                        **t,
                        "text_clean": clean,
                        "asset_tag": asset_choice,
                        "sentiment_score": result["score"],
                        "sentiment_label": result["label"],
                        "model_used": result["model"],
                    }
                    enriched.append(t_enriched)
                    try:
                        save_tweet(t_enriched)
                    except Exception:
                        pass

                signals = compute_all_windows(enriched, asset_choice)

                # Save signals
                for window_key, sig in signals.items():
                    mins = int(window_key.replace("min", ""))
                    try:
                        save_tweet_signal({
                            "timestamp": datetime.utcnow(),
                            "asset": asset_choice,
                            "window_minutes": mins,
                            "signal_value": sig["signal"],
                            "tweet_count": sig["tweet_count"],
                            "alert": sig["alert"],
                        })
                    except Exception:
                        pass

            # Gauges
            st.subheader(f"Signal Gauges — {asset_choice}")
            cols = st.columns(3)
            window_labels = ["1h", "4h", "24h"]
            window_keys = ["60min", "240min", "1440min"]

            for i, (wl, wk) in enumerate(zip(window_labels, window_keys)):
                sig = signals.get(wk, {"signal": 0.0, "tweet_count": 0, "alert": False})
                val = sig["signal"]
                color = "red" if val < -ALERT_THRESHOLD else ("green" if val > ALERT_THRESHOLD else "gray")
                label = (f"\u2b07 BEARISH {asset_choice}" if val < -ALERT_THRESHOLD
                         else f"\u2b06 BULLISH {asset_choice}" if val > ALERT_THRESHOLD
                         else f"\u2192 NEUTRAL")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={"text": f"Signal {wl}"},
                    gauge={
                        "axis": {"range": [-1, 1]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [-1, -ALERT_THRESHOLD], "color": "rgba(255,0,0,0.15)"},
                            {"range": [-ALERT_THRESHOLD, ALERT_THRESHOLD], "color": "rgba(128,128,128,0.1)"},
                            {"range": [ALERT_THRESHOLD, 1], "color": "rgba(0,255,0,0.15)"},
                        ],
                    },
                ))
                fig.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30))
                with cols[i]:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{label} ({sig['tweet_count']} tweets)")
                    if sig["alert"]:
                        st.warning(f"\u26a0 ALERT: |signal| >= {ALERT_THRESHOLD}")

            # Line chart: signal + price
            st.subheader("Signal vs Price (7d)")
            try:
                from shared.backtest.price_client import PriceClient
                pc = PriceClient()
                end = datetime.utcnow()
                start = end - timedelta(days=7)
                prices = pc.get_prices(asset_choice, start, end, horizon_hours=24)

                from shared.db.database import get_tweet_signals
                db_signals = get_tweet_signals(asset_choice)

                if db_signals and not prices.empty:
                    sig_df = pd.DataFrame(db_signals)
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig2.add_trace(
                        go.Scatter(x=sig_df["timestamp"], y=sig_df["signal_value"],
                                   name="Signal", mode="lines+markers"),
                        secondary_y=False,
                    )
                    fig2.add_trace(
                        go.Scatter(x=prices.index, y=prices["close"],
                                   name="Price", mode="lines"),
                        secondary_y=True,
                    )
                    fig2.update_layout(height=400)
                    fig2.update_yaxes(title_text="Signal", secondary_y=False)
                    fig2.update_yaxes(title_text="Price", secondary_y=True)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Not enough data for chart.")
            except Exception as e:
                st.error(f"Price chart error: {e}")

            # Tweet dataframe
            st.subheader("Tweets")
            df = pd.DataFrame([{
                "timestamp": t["created_at"],
                "author": t["author"],
                "tweet": t["text"][:100],
                "score": t["sentiment_score"],
                "label": t["sentiment_label"],
            } for t in enriched[:50]])

            def color_label(val):
                if val == "positive":
                    return "background-color: rgba(0,255,0,0.2)"
                elif val == "negative":
                    return "background-color: rgba(255,0,0,0.2)"
                return ""

            st.dataframe(df.style.map(color_label, subset=["label"]), use_container_width=True)

            # Pie + bar
            col_a, col_b = st.columns(2)
            with col_a:
                labels = df["label"].value_counts()
                fig_pie = go.Figure(go.Pie(labels=labels.index, values=labels.values))
                fig_pie.update_layout(title="Sentiment Distribution", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                from collections import Counter
                all_words = " ".join(df["tweet"]).lower().split()
                word_counts = Counter(w for w in all_words if len(w) > 3)
                top10 = word_counts.most_common(10)
                fig_bar = go.Figure(go.Bar(x=[w[0] for w in top10], y=[w[1] for w in top10]))
                fig_bar.update_layout(title="Top 10 Words", height=300)
                st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Twitter Backtest
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Twitter Signal Backtest")

    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("Start", value=datetime(2024, 1, 1), key="bt_start")
    with col2:
        bt_end = st.date_input("End", value=datetime(2024, 3, 1), key="bt_end")

    horizons_map = {"15min": 0.25, "30min": 0.5, "1h": 1, "4h": 4, "12h": 12,
                    "1d": 24, "3d": 72, "1w": 168}
    horizon_label = st.select_slider("Horizon", options=list(horizons_map.keys()),
                                     value="1d", key="bt_horizon")
    horizon_hours = horizons_map[horizon_label]

    if horizon_hours < 1:
        st.info(f"\U0001f4e1 Source: Alpha Vantage 1min (horizon={horizon_label})")
    elif horizon_hours <= 24:
        st.info(f"\U0001f4c8 Source: yfinance 1h (horizon={horizon_label})")
    else:
        st.info(f"\U0001f4c8 Source: yfinance 1d (horizon={horizon_label})")

    bt_asset = st.selectbox("Asset", list(ASSETS.keys()), key="bt_asset")

    if st.button("\u25b6 Run Backtest", key="bt_run"):
        try:
            with st.spinner("Running backtest..."):
                from shared.db.database import get_tweet_signals
                from shared.backtest.engine import BacktestEngine

                db_signals = get_tweet_signals(bt_asset)
                if not db_signals:
                    st.warning("No signals in DB. Run Twitter Live first with Refresh.")
                else:
                    signals = [{"timestamp": s["timestamp"], "signal_value": s["signal_value"]}
                               for s in db_signals]
                    engine = BacktestEngine()
                    results = engine.run(signals, bt_asset, horizon_hours)

                    if "error" in results:
                        st.warning(results["error"])
                    elif not results:
                        st.warning("No results returned.")
                    else:
                        # KPIs
                        kpi_cols = st.columns(5)
                        kpi_cols[0].metric("Directional Accuracy", f"{results['directional_accuracy']:.2%}")
                        kpi_cols[1].metric("Pearson r", f"{results['pearson_r']:.3f}")
                        kpi_cols[2].metric("Spearman r", f"{results['spearman_r']:.3f}")
                        kpi_cols[3].metric("N Signals", results["n_signals"])
                        kpi_cols[4].metric("p-value", f"{results['pearson_pval']:.4f}")

                        # Forward returns by bucket
                        buckets = results["forward_returns_by_bucket"]
                        fig_buckets = go.Figure(go.Bar(
                            x=list(buckets.keys()), y=list(buckets.values()),
                            marker_color=["red" if v < 0 else "green" for v in buckets.values()]
                        ))
                        fig_buckets.update_layout(title="Forward Returns by Sentiment Bucket", height=350)
                        st.plotly_chart(fig_buckets, use_container_width=True)

                        # Rolling correlation
                        rc = results["rolling_correlation"]
                        if rc:
                            rc_df = pd.DataFrame(rc)
                            fig_rc = go.Figure()
                            fig_rc.add_trace(go.Scatter(
                                x=rc_df["timestamp"], y=rc_df["rolling_corr"],
                                mode="lines", name="Rolling Corr"
                            ))
                            fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig_rc.update_layout(title="Rolling Correlation (20 obs)", height=350)
                            st.plotly_chart(fig_rc, use_container_width=True)

                        # Confusion matrix
                        cm = results["confusion_matrix"]
                        cm_vals = [[cm.get("TP", 0), cm.get("FP", 0)],
                                   [cm.get("FN", 0), cm.get("TN", 0)]]
                        fig_cm = go.Figure(go.Heatmap(
                            z=cm_vals, x=["Predicted +", "Predicted -"],
                            y=["Actual +", "Actual -"],
                            text=[["TP", "FP"], ["FN", "TN"]],
                            texttemplate="%{text}: %{z}",
                            colorscale="Blues",
                        ))
                        fig_cm.update_layout(title="Confusion Matrix", height=350)
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # Scatter
                        raw_df = results.get("raw_df")
                        if raw_df is not None and len(raw_df) > 1:
                            fig_scatter = go.Figure()
                            fig_scatter.add_trace(go.Scatter(
                                x=raw_df["signal_value"], y=raw_df["forward_return"],
                                mode="markers", name="Signals"
                            ))
                            # OLS trendline
                            z = np.polyfit(raw_df["signal_value"], raw_df["forward_return"], 1)
                            x_line = np.linspace(raw_df["signal_value"].min(), raw_df["signal_value"].max(), 50)
                            fig_scatter.add_trace(go.Scatter(
                                x=x_line, y=np.polyval(z, x_line),
                                mode="lines", name="OLS Trend", line=dict(dash="dash", color="red")
                            ))
                            fig_scatter.update_layout(title="Signal vs Forward Return", height=350,
                                                     xaxis_title="Signal", yaxis_title="Forward Return")
                            st.plotly_chart(fig_scatter, use_container_width=True)

                        # Accuracy by quintile
                        abq = results.get("accuracy_by_quintile", {})
                        if abq:
                            fig_abq = go.Figure(go.Bar(x=list(abq.keys()), y=list(abq.values())))
                            fig_abq.update_layout(title="Accuracy by Quintile", height=350)
                            st.plotly_chart(fig_abq, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Document Analyzer
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Document Sentiment Analyzer")

    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        doc_model = st.radio("Model", ["VADER", "FinBERT"], key="doc_model")
        doc_type = st.selectbox("Document Type",
                                ["news", "hedge_fund_letter", "earnings_call", "fomc_minutes", "research_note"],
                                key="doc_type")
        input_mode = st.radio("Input Mode",
                              ["\U0001f4ce Upload PDF", "\U0001f517 URL", "\u270f\ufe0f Free Text", "\U0001f4da Mock Document"],
                              key="doc_input")

    with col_main:
        doc_text = None
        doc_title = None
        doc_source = None
        doc_published = None

        if input_mode == "\U0001f4ce Upload PDF":
            uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_up")
            if uploaded:
                try:
                    from module2_nlp.ingestion.pdf_parser import PdfParser
                    parser = PdfParser()
                    parsed = parser.parse(uploaded.read(), doc_type=doc_type)
                    doc_text = parsed["text"]
                    doc_title = parsed["title"]
                except Exception as e:
                    st.error(f"PDF parse error: {e}")

        elif input_mode == "\U0001f517 URL":
            url = st.text_input("URL", key="doc_url")
            if url and st.button("Fetch", key="doc_fetch"):
                try:
                    import requests
                    from module2_nlp.ingestion.html_parser import HtmlParser
                    with st.spinner("Fetching..."):
                        resp = requests.get(url, timeout=15)
                        parser = HtmlParser()
                        parsed = parser.parse(resp.text, doc_type=doc_type)
                        doc_text = parsed["text"]
                        doc_title = parsed["title"]
                        doc_published = parsed["published_at"]
                except Exception as e:
                    st.error(f"Fetch error: {e}")

        elif input_mode == "\u270f\ufe0f Free Text":
            doc_text = st.text_area("Paste text", height=300, key="doc_text")

        elif input_mode == "\U0001f4da Mock Document":
            from module2_nlp.ingestion.mock_documents import get_mock_documents
            mocks = get_mock_documents()
            mock_options = [f"{d['title']} ({d['source']})" for d in mocks]
            selected = st.selectbox("Select document", mock_options, key="doc_mock")
            idx = mock_options.index(selected)
            doc_text = mocks[idx]["text"]
            doc_title = mocks[idx]["title"]
            doc_source = mocks[idx]["source"]
            doc_type = mocks[idx]["doc_type"]
            doc_published = mocks[idx]["published_at"]

        if st.button("\U0001f50d Analyze", key="doc_analyze") and doc_text:
            try:
                with st.spinner("Analyzing document..."):
                    from module2_nlp.nlp.pipeline import process_document

                    model_key = "finbert" if doc_model == "FinBERT" else "vader"
                    result = process_document(
                        doc_text, file_type="txt", doc_type=doc_type,
                        model=model_key, title=doc_title, source=doc_source,
                        published_at=doc_published
                    )

                signals = result["signals"]
                if not signals:
                    st.warning("No entities detected.")
                else:
                    col_left, col_right = st.columns(2)

                    with col_left:
                        # Entity table
                        sig_rows = []
                        for entity, sig in sorted(signals.items(),
                                                  key=lambda x: abs(x[1]["signal_value"]),
                                                  reverse=True):
                            label = ("positive" if sig["signal_value"] > 0.05
                                     else "negative" if sig["signal_value"] < -0.05
                                     else "neutral")
                            sig_rows.append({
                                "entity": entity,
                                "score": round(sig["signal_value"], 3),
                                "mentions": sig["mention_count"],
                                "label": label,
                            })

                        sig_df = pd.DataFrame(sig_rows)

                        def color_score(val):
                            if val == "positive":
                                return "background-color: rgba(0,255,0,0.2)"
                            elif val == "negative":
                                return "background-color: rgba(255,0,0,0.2)"
                            return ""

                        st.dataframe(sig_df.style.map(color_score, subset=["label"]),
                                     use_container_width=True)

                        # Top 3 alerts
                        st.subheader("Top Alerts")
                        for row in sig_rows[:3]:
                            delta_color = "normal" if row["score"] > 0 else "inverse"
                            st.metric(row["entity"], f"{row['score']:+.3f}",
                                      delta=row["label"], delta_color=delta_color)

                    with col_right:
                        # Bar chart
                        fig_bar = go.Figure(go.Bar(
                            y=[r["entity"] for r in sig_rows],
                            x=[r["score"] for r in sig_rows],
                            orientation="h",
                            marker_color=["green" if r["score"] > 0 else "red" if r["score"] < 0
                                          else "gray" for r in sig_rows],
                        ))
                        fig_bar.update_layout(title="Entity Sentiment", height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Annotated text
                        st.subheader("Annotated Text")
                        annotated = doc_text
                        for row in sig_rows:
                            entity = row["entity"]
                            score = row["score"]
                            color = ("rgba(0,200,0,0.3)" if score > 0.05
                                     else "rgba(200,0,0,0.3)" if score < -0.05
                                     else "rgba(128,128,128,0.2)")
                            # Case-insensitive replace
                            import re
                            pattern = re.compile(re.escape(entity), re.IGNORECASE)
                            annotated = pattern.sub(
                                f'<span style="background-color:{color};padding:2px 4px;border-radius:3px">'
                                f'<b>{entity}</b></span>',
                                annotated
                            )
                        st.markdown(annotated, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Document Backtest
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Document Signal Backtest")

    try:
        from shared.db.database import get_all_entities, get_doc_signals

        entities = get_all_entities()
        if not entities:
            st.info("No entities in DB. Analyze some documents first (Tab 3).")
        else:
            entity_sel = st.selectbox("Entity", entities, key="dbt_entity")
            doc_type_opts = ["all", "news", "hedge_fund_letter", "earnings_call",
                             "fomc_minutes", "research_note"]
            doc_type_sel = st.selectbox("Filter doc_type", doc_type_opts, key="dbt_dtype")

            col1, col2 = st.columns(2)
            with col1:
                dbt_start = st.date_input("Start", value=datetime(2024, 1, 1), key="dbt_start")
            with col2:
                dbt_end = st.date_input("End", value=datetime(2024, 12, 31), key="dbt_end")

            horizons_map_d = {"15min": 0.25, "30min": 0.5, "1h": 1, "4h": 4, "12h": 12,
                              "1d": 24, "3d": 72, "1w": 168}
            dbt_horizon = st.select_slider("Horizon", options=list(horizons_map_d.keys()),
                                           value="1d", key="dbt_horizon")
            dbt_hours = horizons_map_d[dbt_horizon]

            if dbt_hours < 1:
                st.info(f"\U0001f4e1 Source: Alpha Vantage 1min")
            elif dbt_hours <= 24:
                st.info(f"\U0001f4c8 Source: yfinance 1h")
            else:
                st.info(f"\U0001f4c8 Source: yfinance 1d")

            price_assets = ["oil", "gold", "SPX", "NASDAQ", "EUR/USD"]
            price_asset = st.selectbox("Price Asset", price_assets, key="dbt_passet")

            if st.button("\u25b6 Run Backtest", key="dbt_run"):
                try:
                    with st.spinner("Running document backtest..."):
                        dtype = None if doc_type_sel == "all" else doc_type_sel
                        doc_sigs = get_doc_signals(entity=entity_sel, doc_type_filter=dtype)

                        if not doc_sigs:
                            st.warning("No signals found for this entity.")
                        else:
                            signals = [
                                {"timestamp": s.get("published_at", datetime(2024, 6, 1)),
                                 "signal_value": s["signal_value"]}
                                for s in doc_sigs if s.get("published_at") or s.get("signal_value")
                            ]

                            from shared.backtest.engine import BacktestEngine
                            engine = BacktestEngine()
                            results = engine.run(signals, price_asset, dbt_hours)

                            if "error" in results:
                                st.warning(results["error"])
                            elif not results:
                                st.warning("No results.")
                            else:
                                # Same visualization as Tab 2
                                kpi_cols = st.columns(5)
                                kpi_cols[0].metric("Dir. Accuracy", f"{results['directional_accuracy']:.2%}")
                                kpi_cols[1].metric("Pearson r", f"{results['pearson_r']:.3f}")
                                kpi_cols[2].metric("Spearman r", f"{results['spearman_r']:.3f}")
                                kpi_cols[3].metric("N Signals", results["n_signals"])
                                kpi_cols[4].metric("p-value", f"{results['pearson_pval']:.4f}")

                                buckets = results["forward_returns_by_bucket"]
                                fig_b = go.Figure(go.Bar(
                                    x=list(buckets.keys()), y=list(buckets.values()),
                                    marker_color=["red" if v < 0 else "green" for v in buckets.values()]
                                ))
                                fig_b.update_layout(title="Forward Returns by Bucket", height=350)
                                st.plotly_chart(fig_b, use_container_width=True)

                                rc = results["rolling_correlation"]
                                if rc:
                                    rc_df = pd.DataFrame(rc)
                                    fig_rc = go.Figure()
                                    fig_rc.add_trace(go.Scatter(
                                        x=rc_df["timestamp"], y=rc_df["rolling_corr"],
                                        mode="lines"
                                    ))
                                    fig_rc.add_hline(y=0, line_dash="dash")
                                    fig_rc.update_layout(title="Rolling Correlation", height=350)
                                    st.plotly_chart(fig_rc, use_container_width=True)

                                cm = results["confusion_matrix"]
                                cm_vals = [[cm.get("TP", 0), cm.get("FP", 0)],
                                           [cm.get("FN", 0), cm.get("TN", 0)]]
                                fig_cm = go.Figure(go.Heatmap(
                                    z=cm_vals, x=["Pred +", "Pred -"],
                                    y=["Act +", "Act -"],
                                    text=[["TP", "FP"], ["FN", "TN"]],
                                    texttemplate="%{text}: %{z}",
                                    colorscale="Blues",
                                ))
                                fig_cm.update_layout(title="Confusion Matrix", height=350)
                                st.plotly_chart(fig_cm, use_container_width=True)

                                raw_df = results.get("raw_df")
                                if raw_df is not None and len(raw_df) > 1:
                                    fig_s = go.Figure()
                                    fig_s.add_trace(go.Scatter(
                                        x=raw_df["signal_value"], y=raw_df["forward_return"],
                                        mode="markers"
                                    ))
                                    z = np.polyfit(raw_df["signal_value"], raw_df["forward_return"], 1)
                                    x_l = np.linspace(raw_df["signal_value"].min(),
                                                      raw_df["signal_value"].max(), 50)
                                    fig_s.add_trace(go.Scatter(
                                        x=x_l, y=np.polyval(z, x_l),
                                        mode="lines", line=dict(dash="dash", color="red")
                                    ))
                                    fig_s.update_layout(title="Signal vs Return", height=350)
                                    st.plotly_chart(fig_s, use_container_width=True)

                except Exception as e:
                    st.error(f"Backtest error: {e}")

    except Exception as e:
        st.error(f"Error loading entities: {e}")
