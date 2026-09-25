"""Summarize a transcript into structured JSON with Claude (chunked map -> merge for long ones)."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Literal, TypeVar

import anthropic
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Config
from .db import DB
from .models import Episode, Transcript, fmt_ts

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# --- Output schema --------------------------------------------------------------


class Idea(BaseModel):
    idea: str = Field(description="The idea, stated crisply in one or two sentences.")
    why_it_matters: str = Field(description="Why it is non-obvious or useful.")
    timestamp: str | None = Field(description="Transcript timestamp like 12:34 or 1:02:03.")


class Advice(BaseModel):
    advice: str = Field(description="A concrete action someone could take, imperative voice.")
    context: str = Field(description="Who said it and the reasoning behind it.")
    timestamp: str | None = Field(description="Transcript timestamp like 12:34 or 1:02:03.")


class Resource(BaseModel):
    name: str
    kind: Literal["book", "tool", "app", "website", "podcast", "paper", "course", "person", "other"]
    creator: str | None = Field(description="Author / maker, if mentioned or well known.")
    context: str = Field(description="Why it came up / what was said about it.")
    timestamp: str | None


class InterestIdeas(BaseModel):
    interest: str = Field(description="One of the listener's interests, verbatim.")
    ideas: list[Idea] = Field(
        description="Ideas from the episode applicable to this interest. Empty if none."
    )


class Quote(BaseModel):
    quote: str
    speaker: str | None
    timestamp: str | None


class EpisodeSummary(BaseModel):
    one_liner: str = Field(description="One sentence: what this episode is about.")
    guests: list[str]
    summary: list[str] = Field(description="Exactly 5 bullets summarizing the episode.")
    non_obvious_ideas: list[Idea]
    actionable_advice: list[Advice]
    resources: list[Resource] = Field(description="Books, tools and other resources mentioned.")
    interest_ideas: list[InterestIdeas] = Field(
        description="One entry per listener interest, in the order given."
    )
    notable_quotes: list[Quote] = Field(description="Up to 5 memorable verbatim quotes.")
    topics: list[str] = Field(description="3-8 short lowercase topic tags, e.g. 'sleep'.")


# --- Prompts --------------------------------------------------------------------

SYSTEM = """You turn podcast transcripts into structured notes for a busy listener who \
wants the substance without listening.

The listener's interests:
{interests}

Guidelines:
- Ignore sponsor reads, ads, promos and housekeeping entirely; never cite them.
- Prefer specific, surprising, and actionable content over generic truisms. \
"Get more sleep" is not useful; "keep the bedroom at 18°C and stop caffeine by noon" is.
- Timestamps must come from the [h:mm:ss]/[mm:ss] markers in the transcript: use the marker \
of the paragraph where the point is made. If the transcript has no markers, use null.
- For each interest, include only ideas that genuinely apply, and explain the application \
in why_it_matters. For "{first_interest}" think like a practitioner: positioning, \
offers, pricing, lead generation, credibility, delivery, tooling, risk, and specific \
threats or controls worth knowing. An empty list is better than a stretch.
- Attribute ideas and quotes to the right speaker when the transcript makes it clear.
- Quotes must be verbatim from the transcript."""

EPISODE_PROMPT = """Podcast: {podcast}
Episode: {title}
Published: {published}
Description: {description}

<transcript>
{transcript}
</transcript>

Produce the structured notes for this episode."""

CHUNK_PROMPT = """Podcast: {podcast}
Episode: {title}

This is part {i} of {n} of a long transcript (covering {start}–{end}). Extract everything \
noteworthy from THIS part only; another step will merge the parts. For `summary`, give the \
3-5 most important points of this part.

<transcript_part>
{transcript}
</transcript_part>"""

MERGE_PROMPT = """Podcast: {podcast}
Episode: {title}
Description: {description}

Below are structured notes extracted from {n} consecutive parts of this episode's transcript. \
Merge them into one set of notes for the whole episode: exactly 5 summary bullets for the \
whole episode, deduplicate resources/ideas/advice (keep the earliest timestamp), keep only \
the strongest non-obvious ideas and quotes, and keep one interest_ideas entry per interest.

<part_notes>
{parts}
</part_notes>"""


# --- Transcript formatting & chunking -------------------------------------------


def to_paragraphs(t: Transcript, target_seconds: float = 45.0) -> list[tuple[float, str]]:
    """Group segments into ~45s paragraphs, each rendered as '[mm:ss] Speaker: text'."""
    paras: list[tuple[float, str]] = []
    buf: list[str] = []
    start = 0.0
    speaker = None

    def flush():
        if buf:
            prefix = f"[{fmt_ts(start)}] " if t.timed else ""
            who = f"{speaker}: " if speaker else ""
            paras.append((start, f"{prefix}{who}{' '.join(buf)}"))

    for seg in t.segments:
        new_speaker = seg.speaker != speaker and seg.speaker is not None
        too_long = t.timed and buf and seg.start - start >= target_seconds
        untimed_break = not t.timed and buf  # untimed transcripts are already paragraphs
        if not buf or new_speaker or too_long or untimed_break:
            flush()
            buf, start, speaker = [], seg.start, seg.speaker
        buf.append(seg.text.strip())
    flush()
    return paras


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5) + 1  # conservative for English prose


@dataclass
class Chunk:
    start: float
    end: float
    text: str


def chunk_paragraphs(
    paras: list[tuple[float, str]], max_tokens: int, overlap_seconds: float
) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    while i < len(paras):
        size, j = 0, i
        while j < len(paras) and (j == i or size + estimate_tokens(paras[j][1]) <= max_tokens):
            size += estimate_tokens(paras[j][1])
            j += 1
        end = paras[j][0] if j < len(paras) else paras[j - 1][0]
        chunks.append(Chunk(paras[i][0], end, "\n\n".join(p for _, p in paras[i:j])))
        if j >= len(paras):
            break
        # Step back to overlap a little context into the next chunk.
        k = j
        while k - 1 > i and paras[j][0] - paras[k - 1][0] < overlap_seconds:
            k -= 1
        i = k
    return chunks


# --- Claude calls ---------------------------------------------------------------


class SummaryError(RuntimeError):
    pass


class _RetryableOutput(RuntimeError):
    """Output was truncated or invalid; worth one more try."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Summarizer:
    def __init__(self, cfg: Config, db: DB, client: anthropic.Anthropic | None = None):
        self.cfg = cfg
        self.db = db
        self.sc = cfg.summarization
        self.client = client
        if self.client is None and self.sc.backend == "api":
            self.client = anthropic.Anthropic(max_retries=self.sc.max_retries)
        self.usage = Usage()

    def _cost(self, model: str, u) -> float:
        price = self.cfg.price_for(model)
        if price is None:
            log.warning("No price configured for %s; logging $0", model)
            return 0.0
        cw = price.cache_write if price.cache_write is not None else price.input * 1.25
        cr = price.cache_read if price.cache_read is not None else price.input * 0.1
        return (
            (u.input_tokens or 0) * price.input
            + (u.output_tokens or 0) * price.output
            + (getattr(u, "cache_creation_input_tokens", 0) or 0) * cw
            + (getattr(u, "cache_read_input_tokens", 0) or 0) * cr
        ) / 1_000_000

    def call(
        self,
        *,
        system: str,
        prompt: str,
        output_format: type[T],
        stage: str,
        episode_id: str | None,
        model: str | None = None,
    ) -> T:
        model = model or self.sc.model
        retrying = retry(
            retry=retry_if_exception_type(_RetryableOutput),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, max=20),
            reraise=True,
        )
        if self.sc.backend == "claude-code":
            return retrying(self._call_claude_code)(
                system, prompt, output_format, stage, episode_id
            )

        @retrying
        def attempt() -> T:
            kwargs = {}
            if self.sc.effort:
                kwargs["output_config"] = {"effort": self.sc.effort}
            # API errors (429/5xx/connection) are retried with backoff inside the SDK.
            resp = self.client.messages.parse(
                model=model,
                max_tokens=self.sc.max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_format,
                **kwargs,
            )
            cost = self._cost(model, resp.usage)
            self.usage.input_tokens += resp.usage.input_tokens or 0
            self.usage.output_tokens += resp.usage.output_tokens or 0
            self.usage.cost_usd += cost
            self.db.log_cost(
                episode_id=episode_id,
                stage=stage,
                provider="anthropic",
                model=model,
                cost_usd=cost,
                input_tokens=resp.usage.input_tokens or 0,
                output_tokens=resp.usage.output_tokens or 0,
                cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            )
            log.info(
                "Claude %s [%s]: %d in / %d out tokens, $%.4f (request %s)",
                model,
                stage,
                resp.usage.input_tokens,
                resp.usage.output_tokens,
                cost,
                getattr(resp, "_request_id", "?"),
            )
            if resp.stop_reason == "refusal":
                raise SummaryError(f"Claude declined to summarize ({stage})")
            if resp.stop_reason == "max_tokens":
                raise _RetryableOutput(
                    "Output hit max_tokens; raise summarization.max_output_tokens"
                )
            if resp.parsed_output is None:
                raise _RetryableOutput("No parsed output returned")
            return resp.parsed_output

        try:
            return attempt()
        except ValidationError as e:  # schema mismatch surfaced by the SDK parser
            raise SummaryError(f"Invalid structured output ({stage}): {e}") from e

    def _call_claude_code(
        self, system: str, prompt: str, output_format: type[T], stage: str, episode_id: str | None
    ) -> T:
        """One structured call through headless Claude Code, billed to the Claude subscription."""
        exe = shutil.which("claude")
        if not exe:
            raise SummaryError(
                "Claude Code ('claude') is not on PATH. Install it and run `claude` once to log "
                "in, or set summarization.backend: api."
            )
        cmd = [
            exe,
            "-p",
            CLAUDE_CODE_INSTRUCTION,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(output_format.model_json_schema()),
            "--model",
            self.sc.claude_code_model,
            "--permission-mode",
            "dontAsk",
        ]
        # Without ANTHROPIC_API_KEY, `claude -p` uses the subscription login instead of the API.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        t0 = time.time()
        # Run from an empty folder so this repo's CLAUDE.md / settings don't steer the call.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                proc = subprocess.run(
                    cmd,
                    input=f"{system}\n\n{prompt}",
                    capture_output=True,
                    text=True,
                    cwd=tmp,
                    env=env,
                    timeout=self.sc.claude_code_timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise _RetryableOutput(f"claude -p timed out after {e.timeout}s") from e
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-800:]
            raise _RetryableOutput(f"claude -p exited {proc.returncode}: {detail}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise _RetryableOutput(f"claude -p returned non-JSON: {proc.stdout[:300]}") from e
        if data.get("is_error"):
            raise _RetryableOutput(f"claude -p error: {data.get('result') or data}")

        raw = data.get("structured_output")
        if raw is None:  # fall back to JSON in the text result
            raw = _extract_json(data.get("result") or "")
        try:
            parsed = output_format.model_validate(raw)
        except ValidationError as e:
            raise _RetryableOutput(f"Output didn't match the schema: {e}") from e

        usage = data.get("usage") or {}
        estimate = float(data.get("total_cost_usd") or 0.0)
        self.db.log_cost(
            episode_id=episode_id,
            stage=stage,
            provider="claude-code",
            model=self.sc.claude_code_model,
            cost_usd=0.0,  # covered by the subscription
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        )
        log.info(
            "Claude Code (%s) [%s]: done in %.0fs on your subscription (API-equivalent ~$%.3f)",
            self.sc.claude_code_model,
            stage,
            time.time() - t0,
            estimate,
        )
        return parsed

    def summarize(self, ep: Episode, transcript: Transcript) -> EpisodeSummary:
        interests = self.cfg.interests or ["General self-improvement"]
        system = SYSTEM.format(
            interests="\n".join(f"- {i}" for i in interests), first_interest=interests[0]
        )
        paras = to_paragraphs(transcript)
        full_text = "\n\n".join(p for _, p in paras)
        tokens = estimate_tokens(full_text)
        description = _clip(_strip_html(ep.description or ""), 1500)
        published = ep.published.date().isoformat() if ep.published else "unknown"

        if tokens <= self.sc.max_chunk_tokens:
            log.info("Summarizing in one pass (~%d tokens)", tokens)
            return self.call(
                system=system,
                prompt=EPISODE_PROMPT.format(
                    podcast=ep.podcast,
                    title=ep.title,
                    published=published,
                    description=description,
                    transcript=full_text,
                ),
                output_format=EpisodeSummary,
                stage="summarize",
                episode_id=ep.id,
            )

        chunks = chunk_paragraphs(paras, self.sc.max_chunk_tokens, self.sc.chunk_overlap_seconds)
        log.info("Transcript ~%d tokens -> %d chunks", tokens, len(chunks))
        parts = []
        for i, ch in enumerate(chunks, 1):
            part = self.call(
                system=system,
                prompt=CHUNK_PROMPT.format(
                    podcast=ep.podcast,
                    title=ep.title,
                    i=i,
                    n=len(chunks),
                    start=fmt_ts(ch.start),
                    end=fmt_ts(ch.end),
                    transcript=ch.text,
                ),
                output_format=EpisodeSummary,
                stage=f"summarize:{i}/{len(chunks)}",
                episode_id=ep.id,
            )
            parts.append(part.model_dump())
        return self.call(
            system=system,
            prompt=MERGE_PROMPT.format(
                podcast=ep.podcast,
                title=ep.title,
                description=description,
                n=len(parts),
                parts=json.dumps(parts, ensure_ascii=False, indent=1),
            ),
            output_format=EpisodeSummary,
            stage="merge",
            episode_id=ep.id,
        )


CLAUDE_CODE_INSTRUCTION = (
    "The piped input contains your instructions followed by the material to process. "
    "Follow them. Do not use any tools; answer only with the structured output."
)


def _extract_json(text: str) -> object:
    """Parse JSON from a text reply, tolerating ```json fences or surrounding prose."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise _RetryableOutput(f"No JSON in claude -p result: {text[:300]}") from e


def _strip_html(s: str) -> str:
    import html

    return html.unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"
