import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import time


def get_bbc_article_links():
    base_urls = [
        ("https://www.bbc.com/news/world", "world"),
        ("https://www.bbc.com/news/business", "business"),
        ("https://www.bbc.com/news/technology", "tech"),
        ("https://www.bbc.com/sport", "sports"),
        ("https://www.bbc.com/news", "general"),   # ← general LAST
    ]

    seen  = set()
    links = []

    for url, category in base_urls:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            soup = BeautifulSoup(response.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/news/articles/") or href.startswith("/sport/"):
                    full_link = "https://www.bbc.com" + href

                    if full_link not in seen:
                        seen.add(full_link)
                        links.append({
                            "url":      full_link,
                            "category": category
                        })

        except requests.RequestException as e:
            print(f"⚠️  Failed to fetch {url}: {e}")
            continue

    print(f"📰 Unique article links found: {len(links)}")
    return links

def extract_article_data(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text.strip()

        if len(text) < 150:                         # ✅ Fix 3
            print(f"⚠️  Skipping short article: {url}")
            return None

        print(f"✅ Fetched: {url}")
        return {
            "title":        article.title,
            "text":         text,
            "publish_date": article.publish_date,
            "url":          url,
        }

    except Exception as e:
        print(f"❌ Error with {url}: {e}")
        return None


def scrape_bbc():
    links    = get_bbc_article_links()
    articles = []

    for item in links:
        data = extract_article_data(item["url"])

        if data:
            data["category"] = item["category"]  # comes from section page
            articles.append(data)

        time.sleep(1)

    print(f"\n📊 Total articles extracted: {len(articles)}")
    return articles


if __name__ == "__main__":
    data = scrape_bbc()

    if not data:
        print("❌ No data found!")
    else:
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=["url"])
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.len() >= 150]

        print(f"✅ After cleaning: {len(df)} articles")
        df.to_csv("data/bbc_news.csv", index=False)
        print("💾 Saved to data/bbc_news.csv")
        print(df[["title", "category"]].head())