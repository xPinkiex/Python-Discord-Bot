import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from ddgs import DDGS
import bong_tools
import bong_memory_helpers
from llm_utils import _extract_response_text
import user_data


_SUMMARIZE_PROMPT = (
    "Summarize the following web page in 2-3 short sentences. "
    "Be concise and focus on the key point. "
    "If the content is too brief or empty to summarize, say so.\n\n"
    "{content}"
)

_SUMMARIZE_MODEL = ChatOllama(model="gemma3:12b-cloud", temperature=0.3, num_predict=256, keep_alive=-1)


@tool
def web_search(query: str) -> str:
    """Search the web for information. Use this when you need to look up facts, news, or any information you don't know.
    Args:
        query: The search query string.
    """
    if not user_data.has_permission(bong_tools.current_user_id, "llm"):
        return "You don't have permission to use web search. Ask an admin to grant you the llm tag."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return f"{results[0]['title']}: {results[0]['body']}"
    except Exception as e:
        return f"Search error: {e}"


@tool
def youtube_search(query: str) -> str:
    """Search YouTube for videos. Use this when the user wants to find a YouTube video or when you need to find a YouTube URL to download audio from. Available with the llm or music permission tag.
    Args:
        query: The search query string.
    """
    if not (user_data.has_permission(bong_tools.current_user_id, "llm") or user_data.has_permission(bong_tools.current_user_id, "music")):
        return "You don't have permission to search YouTube. Ask an admin to grant you the llm or music tag."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.videos(query, max_results=3))
        if not results:
            return "No YouTube results found."
        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("content", r.get("url", ""))
            if "youtube.com" in url or "youtu.be" in url:
                lines.append(f"- {title}: {url}")
        if not lines:
            return "No YouTube results found."
        return "YouTube results:\n" + "\n".join(lines[:3])
    except Exception as e:
        return f"YouTube search error: {e}"


@tool
def summarize_url(url: str) -> str:
    """Summarize a web page. Use this when someone shares a URL and you want to tell them what it's about, or when you need to look up information from a URL.
    Args:
        url: The full URL to summarize (e.g. "https://example.com/article").
    """
    if not user_data.has_permission(bong_tools.current_user_id, "llm"):
        return "You don't have permission to summarize URLs. Ask an admin to grant you the llm tag."
    import requests
    from lxml import html as lxml_html

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; BongBot/1.0)"}, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch URL: {e}"

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return f"Unsupported content type: {content_type}. Can only summarize HTML or plain text pages."

    try:
        tree = lxml_html.fromstring(resp.text)
    except Exception:
        text = resp.text[:4000]
    else:
        title = tree.findtext(".//title") or ""
        for script in tree.xpath("//script|//style|//noscript"):
            script.getparent().remove(script)
        text = tree.text_content()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        if title:
            text = f"Title: {title}\n\n{text}"
        text = text[:6000]

    if len(text) < 30:
        return "The page doesn't have enough text content to summarize."

    try:
        response = _SUMMARIZE_MODEL.invoke([HumanMessage(content=_SUMMARIZE_PROMPT.format(content=text))])
        summary = _extract_response_text(response).strip()
    except Exception as e:
        return f"Fetched the page but couldn't summarize it: {e}"

    if not summary:
        return "Could not generate a summary."
    return summary


tools = [web_search, youtube_search, summarize_url]