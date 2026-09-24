from podcast_digest.feeds import parse_feed
from podcast_digest.resolve import (
    classify,
    match_episode,
    page_meta,
    parse_apple_ids,
    spotify_show_name,
)

from .conftest import FIXTURES


def test_parse_feed_newest_first_with_transcripts():
    podcast, eps = parse_feed((FIXTURES / "feed.xml").read_bytes(), feed_url="https://f")
    assert podcast == "The Test Show"
    assert [e.guid for e in eps] == ["guid-701", "guid-700"]
    ep = eps[0]
    assert ep.duration_seconds == 5400
    assert eps[1].duration_seconds == 3723
    assert eps[1].title == "#700: Older Episode & Friends"
    assert {t.type for t in ep.transcripts} == {"text/html", "text/vtt"}
    assert ep.image == "https://example.com/show.jpg"
    assert ep.id == eps[0].id and ep.id != eps[1].id


def test_feed_rejects_entity_expansion():
    import defusedxml
    import pytest

    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel>&a;</channel></rss>'
    )
    with pytest.raises(defusedxml.DefusedXmlException):
        parse_feed(bomb)


def test_classify():
    assert classify("https://open.spotify.com/episode/abc?si=1") == "spotify"
    assert classify("https://podcasts.apple.com/us/podcast/x/id863897795?i=1000") == "apple"
    assert classify("https://cdn.example.com/a/b.mp3?token=1") == "audio"
    assert classify("https://rss.art19.com/tim-ferriss-show") == "url"
    assert classify("~/Downloads/ep.m4a") == "file"


def test_parse_apple_ids():
    url = "https://podcasts.apple.com/us/podcast/the-tim-ferriss-show/id863897795?i=1000712345678"
    assert parse_apple_ids(url) == ("863897795", "1000712345678")


def test_spotify_meta():
    page = (
        "<html><head><title>Ep title | Podcast on Spotify</title>"
        '<meta property="og:title" content="#701: Jane Doe &#8212; Consulting"/>'
        '<meta name="description" content="Listen to this episode from The Tim Ferriss Show '
        'on Spotify. Jane talks."/></head></html>'
    )
    meta = page_meta(page)
    assert meta["og:title"] == "#701: Jane Doe — Consulting"
    assert spotify_show_name(meta) == "The Tim Ferriss Show"


def test_match_episode():
    _, eps = parse_feed((FIXTURES / "feed.xml").read_bytes())
    assert match_episode(eps, "Jane Doe — How to Build a Consulting Practice").guid == "guid-701"
    assert match_episode(eps, "completely unrelated title about gardening") is None
