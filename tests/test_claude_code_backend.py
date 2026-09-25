import json
import os
import stat
import sys

import pytest

from podcast_digest.models import Episode
from podcast_digest.summarize import Summarizer

from .test_summarize import make_summary, transcript

FAKE_CLAUDE = """#!{python}
import json, os, sys
args = sys.argv[1:]
stdin = sys.stdin.read()
log = {{"args": args, "cwd": os.getcwd(), "has_key": "ANTHROPIC_API_KEY" in os.environ,
        "stdin_start": stdin[:200]}}
open({log_path!r}, "a").write(json.dumps(log) + "\\n")
mode = {mode!r}
if mode == "fail-once" and not os.path.exists({log_path!r} + ".failed"):
    open({log_path!r} + ".failed", "w").close()
    print("rate limited", file=sys.stderr); sys.exit(1)
out = {{"type": "result", "is_error": False, "total_cost_usd": 0.07,
        "usage": {{"input_tokens": 20000, "output_tokens": 3000}}}}
if mode == "text-only":
    out["result"] = "Here you go:\\n```json\\n" + json.dumps({summary}) + "\\n```"
else:
    out["result"] = ""
    out["structured_output"] = {summary}
print(json.dumps(out))
"""


def install_fake_claude(tmp_path, monkeypatch, mode="ok"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    log_path = str(tmp_path / "calls.jsonl")
    exe.write_text(
        FAKE_CLAUDE.format(
            python=sys.executable, log_path=log_path, mode=mode, summary=make_summary().model_dump()
        )
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-passed")
    return log_path


def calls(log_path):
    return [json.loads(line) for line in open(log_path)]


@pytest.fixture
def cc_cfg(cfg):
    cfg.summarization.backend = "claude-code"
    return cfg


def test_claude_code_backend(cc_cfg, db, tmp_path, monkeypatch):
    log_path = install_fake_claude(tmp_path, monkeypatch)
    ep = Episode(title="T", podcast="P", guid="g")
    out = Summarizer(cc_cfg, db).summarize(ep, transcript(20))
    assert out.guests == ["Jane Doe"]

    (call,) = calls(log_path)
    args = call["args"]
    assert args[0] == "-p" and "--json-schema" in args and "--output-format" in args
    assert "--model" not in args  # default: Claude Code picks the model
    assert json.loads(args[args.index("--json-schema") + 1])["title"] == "EpisodeSummary"
    assert not call["has_key"]  # subscription, not the API key
    assert "podcasts" not in call["cwd"] or call["cwd"].startswith("/tmp")
    assert call["stdin_start"].startswith("You turn podcast transcripts")

    (row,) = db.cost_report()
    assert row["provider"] == "claude-code" and row["cost_usd"] == 0
    assert row["input_tokens"] == 20000


def test_claude_code_text_fallback_and_retry(cc_cfg, db, tmp_path, monkeypatch):
    import tenacity

    monkeypatch.setattr(tenacity.nap, "sleep", lambda s: None)
    log_path = install_fake_claude(tmp_path, monkeypatch, mode="fail-once")
    out = Summarizer(cc_cfg, db).summarize(Episode(title="T", podcast="P"), transcript(5))
    assert out.one_liner == "An episode." and len(calls(log_path)) == 2


def test_claude_code_json_in_text(cc_cfg, db, tmp_path, monkeypatch):
    install_fake_claude(tmp_path, monkeypatch, mode="text-only")
    out = Summarizer(cc_cfg, db).summarize(Episode(title="T", podcast="P"), transcript(5))
    assert out.topics == ["consulting", "security"]


def test_missing_claude_is_a_clear_error(cc_cfg, db, monkeypatch):
    from podcast_digest.summarize import SummaryError

    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(SummaryError, match="not on PATH"):
        Summarizer(cc_cfg, db).summarize(Episode(title="T", podcast="P"), transcript(5))


def test_model_error_fails_fast(cc_cfg, db, tmp_path, monkeypatch):
    from podcast_digest.summarize import SummaryError

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    count = tmp_path / "count"
    exe.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(count)!r}, 'a').write('x')\n"
        "print(json.dumps({'is_error': True, 'result': 'There is an issue with the selected "
        "model (claude-sonnet-4-6). It may not exist or you may not have access to it.'}))\n"
        "sys.exit(1)\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(SummaryError, match="selected model"):
        Summarizer(cc_cfg, db).summarize(Episode(title="T", podcast="P"), transcript(5))
    assert count.read_text() == "x"  # no retries
