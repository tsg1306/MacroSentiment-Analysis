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
                        "chinese","trade war","tariff","em","developing"],
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
    "we buy","we sell","recommend buy","recommend sell",
    "buy rating","sell rating","strong buy","strong sell",
    "overweight","underweight",
    "go long","go short","remain long","remain short",
    "target price","price target"," tp ",
    "upgrade to","downgrade to","upgrade from","downgrade from",
    "add to position","trim position","initiate coverage",
    "reiterate buy","reiterate sell","reiterate overweight",
    "outperform","market perform","underperform",
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
    domains = [t for t, s in theme_scores.items() if s >= THEME_THRESHOLD]
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
