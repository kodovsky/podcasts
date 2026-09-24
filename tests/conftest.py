from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_digest.config import Config
from podcast_digest.db import DB

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        output_dir=tmp_path / "vault",
        data_dir=tmp_path / "data",
        interests=["Building a solo AI agent security consulting practice", "Health"],
    )


@pytest.fixture
def db(cfg) -> DB:
    d = DB(cfg.db_path)
    yield d
    d.close()


class FakeMessages:
    """Stands in for client.messages; returns queued parsed outputs with fake usage."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, SimpleNamespace):
            return out
        return SimpleNamespace(
            parsed_output=out,
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=10_000,
                output_tokens=2_000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            _request_id="req_test",
        )


class FakeClient:
    def __init__(self, outputs):
        self.messages = FakeMessages(outputs)


@pytest.fixture
def fake_client():
    return FakeClient
