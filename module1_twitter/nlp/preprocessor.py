import re

SYNONYMS = {
    "crude oil": "oil",
    "petroleum": "oil",
    "xau/usd": "gold",
    "federal reserve": "fed",
}

# Pre-compile patterns
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9\s]")
_SPACES_RE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    """Clean a tweet through a deterministic, idempotent pipeline.

    Pipeline: lowercase -> strip URLs -> strip @mentions (keep text) ->
    hashtags without # -> normalize synonyms -> strip special chars ->
    normalize spaces.

    Must be idempotent: clean_tweet(clean_tweet(x)) == clean_tweet(x)
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Strip URLs
    text = _URL_RE.sub("", text)

    # 3. Strip @mentions (remove the @ and username entirely)
    text = _MENTION_RE.sub("", text)

    # 4. Hashtags: remove # but keep the word
    text = _HASHTAG_RE.sub(r"\1", text)

    # 5. Normalize synonyms (must happen before stripping special chars
    #    because some synonyms contain special chars like "/")
    for phrase, replacement in SYNONYMS.items():
        text = text.replace(phrase, replacement)

    # 6. Strip special characters (keep letters, digits, spaces)
    text = _SPECIAL_RE.sub(" ", text)

    # 7. Normalize whitespace
    text = _SPACES_RE.sub(" ", text).strip()

    return text
