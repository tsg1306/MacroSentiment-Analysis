import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


def fit_tweet_topics(texts: list, n_topics: int = 8) -> tuple:
    """
    Fit BERTopic on tweet texts using KMeans clustering (hdbscan unavailable).
    Returns (topic_model, topics_list, topic_labels_dict).
    topic_labels_dict: {topic_id: human_readable_label}
    """
    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    # Use KMeans since hdbscan has no wheel for Python 3.14/Windows
    cluster_model = KMeans(n_clusters=min(n_topics, len(texts) // 10), n_init='auto', random_state=42)

    topic_model = BERTopic(
        embedding_model=model,
        hdbscan_model=cluster_model,
        language="english",
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Build human-readable labels from top keywords
    topic_labels = {}
    try:
        info = topic_model.get_topic_info()
        for _, row in info.iterrows():
            tid = row["Topic"]
            kws = topic_model.get_topic(tid)
            if kws:
                label = " / ".join([w for w, _ in kws[:3]])
                topic_labels[tid] = label
            else:
                topic_labels[tid] = f"Topic {tid}"
    except Exception:
        unique_topics = list(set(topics))
        topic_labels = {t: f"Topic {t}" for t in unique_topics}

    return topic_model, topics, topic_labels


def get_topic_barchart(topic_model: BERTopic, top_n: int = 8):
    """Returns Plotly figure: top keywords per topic."""
    try:
        return topic_model.visualize_barchart(top_n_topics=top_n)
    except Exception:
        return None


def get_topic_map(topic_model: BERTopic):
    """Returns Plotly figure: 2D topic map."""
    try:
        return topic_model.visualize_topics()
    except Exception:
        return None
