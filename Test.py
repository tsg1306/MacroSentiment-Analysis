from module1_twitter.twitter.client import TwitterClient
from module1_twitter.nlp.preprocessor import clean_tweet
from module1_twitter.signal.extractor import compute_all_windows
from shared.nlp.vader_sentiment import VaderSentiment
from config import ASSETS

client = TwitterClient('mock')
vader = VaderSentiment()
enriched = []
for asset, cfg in ASSETS.items():
    for t in client.search(cfg['keywords'], limit=50):
        clean = clean_tweet(t['text'])
        r = vader.analyze(clean)
        enriched.append({**t, 'asset_tag': asset, 'sentiment_score': r['score'], 'sentiment_label': r['label']})

signals = compute_all_windows(enriched, 'WTI')
for k, v in signals.items():
    print(f'WTI {k}: {v["signal"]:+.4f} ({v["tweet_count"]} tweets)')