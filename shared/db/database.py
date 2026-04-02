import os
import sys
from datetime import datetime, timedelta

from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, DateTime, Boolean,
    ForeignKey, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure config is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import DB_PATH

Base = declarative_base()
_engine = None
_Session = None


def _get_engine():
    global _engine, _Session
    if _engine is None:
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        _Session = sessionmaker(bind=_engine)
    return _engine


def get_session():
    _get_engine()
    return _Session()


# ── Models ────────────────────────────────────────────────────────────────────

class Tweet(Base):
    __tablename__ = "tweets"
    id = Column(Text, primary_key=True)
    created_at = Column(DateTime)
    author = Column(Text)
    followers_count = Column(Integer)
    text = Column(Text)
    text_clean = Column(Text)
    retweet_count = Column(Integer)
    like_count = Column(Integer)
    asset_tag = Column(Text)
    sentiment_score = Column(Float)
    sentiment_label = Column(Text)
    model_used = Column(Text)
    inserted_at = Column(DateTime, default=datetime.utcnow)


class TweetSignal(Base):
    __tablename__ = "tweet_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    asset = Column(Text)
    window_minutes = Column(Integer)
    signal_value = Column(Float)
    tweet_count = Column(Integer)
    alert = Column(Boolean)


class Document(Base):
    __tablename__ = "documents"
    id = Column(Text, primary_key=True)
    title = Column(Text)
    source = Column(Text)
    doc_type = Column(Text)
    published_at = Column(DateTime, nullable=True)
    inserted_at = Column(DateTime, default=datetime.utcnow)
    char_count = Column(Integer)


class EntitySentiment(Base):
    __tablename__ = "entity_sentiments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Text, ForeignKey("documents.id"))
    entity = Column(Text)
    entity_type = Column(Text)
    context_text = Column(Text)
    sentiment_score = Column(Float)
    sentiment_label = Column(Text)
    model_used = Column(Text)
    confidence = Column(Float)


class DocumentSignal(Base):
    __tablename__ = "document_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Text, ForeignKey("documents.id"))
    entity = Column(Text)
    signal_value = Column(Float)
    mention_count = Column(Integer)
    alert = Column(Boolean)
    published_at = Column(DateTime, nullable=True)


class CorpusIngested(Base):
    __tablename__ = "corpus_ingested"
    filename = Column(Text, primary_key=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)


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


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    engine = _get_engine()
    Base.metadata.create_all(engine)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def save_tweet(tweet_dict):
    """INSERT OR IGNORE — idempotent on PK."""
    session = get_session()
    try:
        existing = session.get(Tweet, tweet_dict["id"])
        if existing is None:
            session.add(Tweet(**tweet_dict))
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_tweet_signal(signal_dict):
    session = get_session()
    try:
        session.add(TweetSignal(**signal_dict))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_document(doc_dict):
    """INSERT OR IGNORE — idempotent on SHA256 PK."""
    session = get_session()
    try:
        existing = session.get(Document, doc_dict["id"])
        if existing is None:
            session.add(Document(**doc_dict))
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_entity_sentiment(record_dict):
    session = get_session()
    try:
        session.add(EntitySentiment(**record_dict))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_doc_signal(signal_dict):
    session = get_session()
    try:
        session.add(DocumentSignal(**signal_dict))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_tweets(asset=None, since_minutes=None):
    session = get_session()
    try:
        q = session.query(Tweet)
        if asset:
            q = q.filter(Tweet.asset_tag == asset)
        if since_minutes:
            cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
            q = q.filter(Tweet.created_at >= cutoff)
        return [
            {c.name: getattr(row, c.name) for c in Tweet.__table__.columns}
            for row in q.all()
        ]
    finally:
        session.close()


def get_tweet_signals(asset=None):
    session = get_session()
    try:
        q = session.query(TweetSignal)
        if asset:
            q = q.filter(TweetSignal.asset == asset)
        return [
            {c.name: getattr(row, c.name) for c in TweetSignal.__table__.columns}
            for row in q.order_by(TweetSignal.timestamp).all()
        ]
    finally:
        session.close()


def get_doc_signals(entity=None, doc_type_filter=None):
    session = get_session()
    try:
        q = session.query(DocumentSignal)
        if entity:
            q = q.filter(DocumentSignal.entity == entity)
        if doc_type_filter:
            doc_ids = [
                d.id for d in session.query(Document).filter(
                    Document.doc_type == doc_type_filter
                ).all()
            ]
            q = q.filter(DocumentSignal.document_id.in_(doc_ids))
        return [
            {c.name: getattr(row, c.name) for c in DocumentSignal.__table__.columns}
            for row in q.all()
        ]
    finally:
        session.close()


def get_all_entities():
    session = get_session()
    try:
        rows = session.query(DocumentSignal.entity).distinct().all()
        return [r[0] for r in rows]
    finally:
        session.close()


def get_ingested_files() -> set:
    """Returns set of filenames already processed by ingest_corpus.py."""
    session = get_session()
    try:
        rows = session.query(CorpusIngested.filename).all()
        return {r[0] for r in rows}
    finally:
        session.close()


def mark_file_ingested(filename: str):
    """Mark a PDF filename as ingested. Idempotent."""
    session = get_session()
    try:
        existing = session.get(CorpusIngested, filename)
        if existing is None:
            session.add(CorpusIngested(filename=filename))
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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
        q = session.query(DocumentAnalysis, Document).outerjoin(
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
                "doc_id":         da.doc_id,
                "title":          doc.title if doc else None,
                "source":         doc.source if doc else None,
                "doc_type":       doc.doc_type if doc else None,
                "published_at":   doc.published_at if doc else None,
                "domains":        domains,
                "primary_domain": da.primary_domain,
                "stance":         da.stance,
                "trade_signal":   da.trade_signal,
                "explicit_count": da.explicit_count,
                "implicit_count": da.implicit_count,
                "summary_json":   da.summary_json,
            })
        return results
    finally:
        session.close()
