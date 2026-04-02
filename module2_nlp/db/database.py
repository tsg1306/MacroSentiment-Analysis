import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.db.database import (
    init_db, save_document, save_entity_sentiment, save_doc_signal,
    get_doc_signals, get_all_entities
)
