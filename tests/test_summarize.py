from types import SimpleNamespace

import pytest

from podcast_digest.models import Episode, Segment, Transcript
from podcast_digest.summarize import (
    EpisodeSummary,
    Summarizer,
    SummaryError,
    chunk_paragraphs,
    estimate_tokens,
    to_paragraphs,
)


def make_summary(**kw) -> EpisodeSummary:
    base = dict(
        one_liner="An episode.",
        guests=["Jane Doe"],
        summary=[f"Point {i}" for i in range(5)],
        non_obvious_ideas=[{"idea": "Niche down", "why_it_matters": "Trust", "timestamp": "01:05"}],
        actionable_advice=[{"advice": "Write a teardown", "context": "Jane", "timestamp": "02:00"}],
        resources=[
            {
                "name": "The E-Myth",
                "kind": "book",
                "creator": "Michael Gerber",
                "context": "Systems",
                "timestamp": "03:00",
            }
        ],
        interest_ideas=[
            {
                "interest": "Building a solo AI agent security consulting practice",
                "ideas": [
                    {
                        "idea": "Offer fixed-scope audits",
                        "why_it_matters": "Easy yes",
                        "timestamp": "04:00",
                    }
                ],
            },
            {"interest": "Health", "ideas": []},
        ],
        notable_quotes=[{"quote": "Own it.", "speaker": "Jane Doe", "timestamp": "01:05"}],
        topics=["consulting", "security"],
    )
    base.update(kw)
    return EpisodeSummary.model_validate(base)


def transcript(minutes: int, words_per_seg: int = 20) -> Transcript:
    segs = [
        Segment(
            i * 10.0,
            i * 10.0 + 10,
            " ".join(["word"] * words_per_seg),
            "Tim" if (i // 30) % 2 == 0 else "Jane",
        )
        for i in range(minutes * 6)
    ]
    return Transcript(segs, source="mlx-whisper")


def test_paragraphs_have_timestamps_and_speakers():
    paras = to_paragraphs(transcript(10))
    assert paras[0][1].startswith("[00:00] Tim: word")
    assert all(p[0] < 600 for p in paras)
    # new paragraph at least every 45s, and on speaker change at 5:00
    assert any(p[1].startswith("[05:00] Jane:") for p in paras)


def test_untimed_has_no_markers():
    t = Transcript(
        [Segment(0, 0, "Para one."), Segment(0, 0, "Para two.")], source="published", timed=False
    )
    assert [p for _, p in to_paragraphs(t)] == ["Para one.", "Para two."]


def test_chunking_covers_everything_with_overlap():
    paras = to_paragraphs(transcript(180))
    total = sum(estimate_tokens(p) for _, p in paras)
    chunks = chunk_paragraphs(paras, max_tokens=total // 4, overlap_seconds=60)
    assert 4 <= len(chunks) <= 6
    assert chunks[0].start == 0 and chunks[-1].end == paras[-1][0]
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start < a.end  # overlap
        assert a.end - b.start <= 120
    joined = "\n\n".join(c.text for c in chunks)
    assert all(p in joined for _, p in paras)


def test_single_pass(cfg, db, fake_client):
    client = fake_client([make_summary()])
    s = Summarizer(cfg, db, client=client)
    ep = Episode(title="T", podcast="P", guid="g")
    out = s.summarize(ep, transcript(30))
    assert out.guests == ["Jane Doe"]
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["output_format"] is EpisodeSummary
    assert call["output_config"] == {"effort": "medium"}
    assert "sponsor" in call["system"] and "security consulting" in call["system"]
    assert "[00:00] Tim:" in call["messages"][0]["content"]
    # 10k in * $2/M + 2k out * $10/M = $0.04
    assert db.episode_cost(ep.id) == pytest.approx(0.04)


def test_chunked_then_merged(cfg, db, fake_client):
    cfg.summarization.max_chunk_tokens = 5_000
    t = transcript(90)
    n_chunks = len(chunk_paragraphs(to_paragraphs(t), 5_000, 60))
    client = fake_client([make_summary()] * n_chunks + [make_summary(one_liner="Merged")])
    out = Summarizer(cfg, db, client=client).summarize(Episode(title="T", podcast="P"), t)
    assert out.one_liner == "Merged"
    assert len(client.messages.calls) == n_chunks + 1
    assert "part 1 of" in client.messages.calls[0]["messages"][0]["content"]
    assert "<part_notes>" in client.messages.calls[-1]["messages"][0]["content"]
    stages = [r["stage"] for r in db.cost_report()]
    assert "merge" in stages and f"summarize:1/{n_chunks}" in stages


def test_max_tokens_is_retried(cfg, db, fake_client, monkeypatch):
    truncated = SimpleNamespace(
        parsed_output=None,
        stop_reason="max_tokens",
        _request_id="r",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = fake_client([truncated, make_summary()])
    import tenacity

    monkeypatch.setattr(tenacity.nap, "sleep", lambda s: None)
    out = Summarizer(cfg, db, client=client).summarize(
        Episode(title="T", podcast="P"), transcript(5)
    )
    assert out.one_liner == "An episode."
    assert len(client.messages.calls) == 2


def test_refusal_raises(cfg, db, fake_client):
    refused = SimpleNamespace(
        parsed_output=None,
        stop_reason="refusal",
        _request_id="r",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    with pytest.raises(SummaryError):
        Summarizer(cfg, db, client=fake_client([refused])).summarize(
            Episode(title="T", podcast="P"), transcript(5)
        )


def test_schema_is_strict_compatible():
    from anthropic.lib._parse._transform import transform_schema  # SDK's own transformer

    schema = transform_schema(EpisodeSummary.model_json_schema())
    assert schema["additionalProperties"] is False
