import feedparser
import pandas as pd
from newspaper import Article
import time


# ✅ Active RSS feeds with correct categories
RSS_FEEDS = {
    "business":  "https://finance.yahoo.com/rss/topstories",
    "tech":      "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,GOOGL,MSFT&region=US&lang=en-US",
    "world":     "https://news.yahoo.com/rss/world",
    "finance":   "https://finance.yahoo.com/rss/headline?s=^GSPC,^DJI",
}

ARTICLES_PER_FEED = 30


def scrape_feed(feed_url: str, category: str) -> list:
    feed     = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries[:ARTICLES_PER_FEED]:
        url = entry.get("link", "")
        if not url:
            continue

        try:
            article = Article(url)
            article.download()
            article.parse()

            text = article.text.strip()
            if len(text) < 150:
                print(f"⚠️  Skipping short article: {url}")
                continue

            articles.append({
                "title":    article.title,
                "text":     text,
                "url":      url,
                "category": category,
            })
            print(f"✅ [{category}] {article.title[:60]}")

        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")

        time.sleep(1)

    return articles


def scrape_yahoo_finance() -> list:
    all_articles = []

    for category, feed_url in RSS_FEEDS.items():
        print(f"\n📡 Scraping feed: {category}")
        articles = scrape_feed(feed_url, category)
        all_articles.extend(articles)
        print(f"   → Got {len(articles)} articles")

    print(f"\n📊 Total articles: {len(all_articles)}")
    return all_articles


if __name__ == "__main__":
    data = scrape_yahoo_finance()

    if not data:
        print("❌ No data scraped!")
    else:
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=["url"])
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.len() >= 150]

        print(f"✅ After cleaning: {len(df)} articles")
        df.to_csv("data/reuters_news.csv", index=False)
        print("💾 Saved to data/reuters_news.csv")
        print(df["category"].value_counts())