from typing import List, Dict, Any
import re
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


_TRASH_PATTERNS = [
    r"cookie",
    r"consent",
    r"privacy policy",
    r"политика конфиденциальности",
    r"использу(ем|ете) cookie",
    r"файлы cookie",
    r"подпис(ать|к)а",
    r"подпишитесь",
    r"реклама",
    r"advertis",
    r"accept all",
    r"agree",
]

def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) >= 40]

    trash_re = re.compile("|".join(_TRASH_PATTERNS), flags=re.IGNORECASE)
    lines = [ln for ln in lines if not trash_re.search(ln)]

    text = "\n".join(lines)

    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


def _remove_noise_tags(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "button", "input"]):
        tag.decompose()

    for tag in soup.find_all(["nav", "footer", "header", "aside"]):
        tag.decompose()


def _text_len(node) -> int:
    if not node:
        return 0
    txt = node.get_text(" ", strip=True)
    return len(txt)


def _link_density(node) -> float:
    if not node:
        return 1.0
    text = node.get_text(" ", strip=True)
    if not text:
        return 1.0
    link_text = " ".join(a.get_text(" ", strip=True) for a in node.find_all("a"))
    return min(1.0, len(link_text) / max(1, len(text)))


def _pick_main_content(soup: BeautifulSoup):
    for sel in ["article", "main"]:
        node = soup.select_one(sel)
        if node and _text_len(node) >= 800:
            return node

    candidates = soup.find_all(["div", "section"], limit=2000)
    scored = []
    for c in candidates:
        cid = " ".join(filter(None, [c.get("id", ""), " ".join(c.get("class", []))])).lower()
        if any(k in cid for k in ["content", "article", "post", "entry", "text", "body", "main"]):
            tl = _text_len(c)
            if tl >= 800:
                ld = _link_density(c)
                score = tl * (1.0 - ld)
                scored.append((score, c))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    best = None
    best_score = 0.0
    for c in candidates:
        tl = _text_len(c)
        if tl < 800:
            continue
        ld = _link_density(c)
        score = tl * (1.0 - ld)
        if score > best_score:
            best_score = score
            best = c

    return best or soup.body or soup


@tool
def web_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a web search (DuckDuckGo), fetch pages, extract main readable text (without HTML мусора).
    Returns: list of dicts: {url, title, text, excerpt, char_count, error}
    """
    search = DuckDuckGoSearchResults(output_format="list", max_results=10)
    results = search.invoke(query)

    links = []
    for r in results:
        link = r.get("link")
        if link and link.startswith(("http://", "https://")):
            links.append(link)

    out: List[Dict[str, Any]] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    }

    for url in links:
        item: Dict[str, Any] = {"url": url, "title": None, "text": "", "excerpt": "", "char_count": 0, "error": None}
        try:
            resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
            ctype = resp.headers.get("Content-Type", "").lower()

            if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                item["error"] = f"Skipped non-HTML content-type: {ctype or 'unknown'}"
                out.append(item)
                continue

            if resp.status_code != 200:
                item["error"] = f"HTTP {resp.status_code}"
                out.append(item)
                continue

            html = resp.text
            soup = BeautifulSoup(html, "lxml") 
            _remove_noise_tags(soup)

            title_tag = soup.find("title")
            if title_tag and title_tag.string:
                item["title"] = title_tag.string.strip()

            main = _pick_main_content(soup)
            raw_text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)

            cleaned = _clean_text(raw_text)

            max_chars = 9000
            cleaned = cleaned[:max_chars].strip()

            item["text"] = cleaned
            item["char_count"] = len(cleaned)
            item["excerpt"] = cleaned[:800] + ("…" if len(cleaned) > 800 else "")

        except Exception as e:
            item["error"] = f"{type(e).__name__}: {e}"

        out.append(item)

    return out