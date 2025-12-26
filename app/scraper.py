import requests
from bs4 import BeautifulSoup
from newspaper import Article

MIN_LEN = 300


def scrape_with_newspaper(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text) >= MIN_LEN else ""
    except Exception:
        return ""


def scrape_with_bs4(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120 Safari/537.36"
            )
        }
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text if len(text) >= MIN_LEN else ""
    except Exception:
        return ""


def scrape_article(url: str) -> str:
    text = scrape_with_newspaper(url)
    if text:
        return text
    return scrape_with_bs4(url)
