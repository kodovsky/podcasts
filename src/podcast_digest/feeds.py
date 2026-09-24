"""RSS feed fetching and parsing (including Podcasting 2.0 <podcast:transcript>)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET

from .http import get
from .models import Episode, TranscriptLink

log = logging.getLogger(__name__)

NS = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "podcast": "https://podcastindex.org/namespace/1.0",
    "content": "http://purl.org/rss/1.0/modules/content/",
}
# Some feeds use the older http:// or a trailing-slash variant of the podcast namespace.
PODCAST_NS_VARIANTS = [
    "https://podcastindex.org/namespace/1.0",
    "http://podcastindex.org/namespace/1.0",
    "https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/1.0.md",
]


def _text(el: Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    return el.text.strip() or None


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _parse_duration(s: str | None) -> float | None:
    if not s:
        return None
    try:
        parts = [float(p) for p in s.strip().split(":")]
    except ValueError:
        return None
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def parse_feed(xml: bytes, feed_url: str | None = None) -> tuple[str, list[Episode]]:
    """Return (podcast title, episodes newest-first)."""
    root = ET.fromstring(xml)
    channel = root.find("channel")
    if channel is None:
        raise ValueError("Not an RSS feed (no <channel>)")
    podcast = _text(channel.find("title")) or "Unknown podcast"
    channel_image = channel.find("itunes:image", NS)
    default_image = channel_image.get("href") if channel_image is not None else None

    episodes: list[Episode] = []
    for item in channel.findall("item"):
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url") if enclosure is not None else None
        transcripts = []
        for ns in PODCAST_NS_VARIANTS:
            for t in item.findall(f"{{{ns}}}transcript"):
                if t.get("url"):
                    transcripts.append(
                        TranscriptLink(
                            url=t.get("url"), type=t.get("type", ""), language=t.get("language")
                        )
                    )
        image_el = item.find("itunes:image", NS)
        description = _text(item.find("content:encoded", NS)) or _text(item.find("description"))
        episodes.append(
            Episode(
                title=_text(item.find("title")) or "Untitled",
                podcast=podcast,
                audio_url=audio_url,
                guid=_text(item.find("guid")) or audio_url,
                feed_url=feed_url,
                link=_text(item.find("link")),
                published=_parse_date(_text(item.find("pubDate"))),
                duration_seconds=_parse_duration(_text(item.find("itunes:duration", NS))),
                description=description,
                image=image_el.get("href") if image_el is not None else default_image,
                transcripts=transcripts,
            )
        )
    episodes.sort(key=lambda e: e.published or datetime.min.replace(tzinfo=UTC), reverse=True)
    return podcast, episodes


def fetch_feed(url: str) -> tuple[str, list[Episode]]:
    log.info("Fetching feed %s", url)
    resp = get(url)
    return parse_feed(resp.content, feed_url=url)


def looks_like_feed(content: bytes) -> bool:
    head = content[:2048].lower()
    return b"<rss" in head or b"<channel" in head
