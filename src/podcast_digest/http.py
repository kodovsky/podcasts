"""Shared HTTP helpers with retries."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

USER_AGENT = "podcast-digest/0.1 (+https://github.com/kodovsky/podcasts)"
TIMEOUT = httpx.Timeout(30.0, read=120.0)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429) or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


_retry = retry(
    retry=retry_if_exception(_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, max=30),
    reraise=True,
)


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)


@_retry
def get(url: str, **params) -> httpx.Response:
    with client() as c:
        resp = c.get(url, params=params or None)
        resp.raise_for_status()
        return resp


@_retry
def download(url: str, dest: Path) -> Path:
    """Stream a (possibly large) file to disk; skips if it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Audio already downloaded: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s", url)
    with client() as c, c.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 16):
                f.write(chunk)
    tmp.rename(dest)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest
