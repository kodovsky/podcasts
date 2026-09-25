"""Configuration loading (YAML -> validated pydantic models)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Engine = Literal["mlx-whisper", "faster-whisper", "deepgram"]


class FeedConfig(BaseModel):
    name: str
    url: str
    since: date | None = None  # ignore episodes published before this date
    max_new_per_run: int = 3


class TranscriptionConfig(BaseModel):
    # "auto" picks mlx-whisper on Apple Silicon, else faster-whisper, else Deepgram.
    engine: Literal["auto"] | Engine = "auto"
    fallback: Engine | None = "deepgram"
    language: str | None = None  # e.g. "en"; None lets Whisper detect it
    mlx_model: str = "mlx-community/whisper-large-v3-turbo"
    faster_whisper_model: str = "large-v3-turbo"
    faster_whisper_device: str = "auto"
    faster_whisper_compute_type: str = "default"
    deepgram_model: str = "nova-3"
    deepgram_usd_per_minute: float = 0.0043  # check against your Deepgram plan
    # Delete downloaded audio once transcribed (files you pass in yourself are never deleted).
    keep_audio: bool = True
    # Use a transcript published in the feed (<podcast:transcript>) when it has timestamps.
    prefer_published: bool = True
    # Also accept published transcripts without timestamps (HTML/plain text).
    allow_untimed_published: bool = False


class ModelPrice(BaseModel):
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens
    cache_write: float | None = None  # defaults to 1.25x input
    cache_read: float | None = None  # defaults to 0.1x input


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-sonnet-5": ModelPrice(input=2.0, output=10.0),
    "claude-opus-5": ModelPrice(input=5.0, output=25.0),
    "claude-opus-5-5": ModelPrice(input=4.0, output=20.0),
    "claude-haiku-4-5": ModelPrice(input=1.0, output=5.0),
}


class SummarizationConfig(BaseModel):
    # api: Claude API (needs ANTHROPIC_API_KEY, pay per token).
    # claude-code: headless Claude Code (`claude -p`) on your Claude subscription.
    backend: Literal["api", "claude-code"] = "api"
    model: str = "claude-sonnet-5"
    # Passed to `claude --model` (e.g. claude-sonnet-5); None uses Claude Code's default model.
    claude_code_model: str | None = None
    claude_code_timeout: int = 1200  # seconds per call
    digest_model: str | None = None  # defaults to `model`
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    max_output_tokens: int = 16000
    # Transcripts longer than this (estimated tokens) are summarized in chunks, then merged.
    max_chunk_tokens: int = 80_000
    chunk_overlap_seconds: int = 60
    max_retries: int = 5  # SDK retries on 429/5xx/connection errors with backoff
    prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICES))

    @property
    def model_label(self) -> str:
        if self.backend == "claude-code":
            return f"claude-code/{self.claude_code_model or 'default'}"
        return self.model


class OutputConfig(BaseModel):
    wikilink_resources: bool = True  # [[Book Title]] links for books/tools
    write_transcript_notes: bool = True  # separate linked transcript note per episode
    tags: list[str] = Field(default_factory=lambda: ["podcast"])


class Config(BaseModel):
    output_dir: Path = Path("./notes")
    data_dir: Path = Path("./data")
    feeds: list[FeedConfig] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("output_dir", "data_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def models_dir(self) -> Path:
        """Whisper model downloads (Hugging Face cache), kept inside the project."""
        return self.data_dir / "models"

    def price_for(self, model: str) -> ModelPrice | None:
        return self.summarization.prices.get(model)


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy config.example.yaml to config.yaml and edit it."
        )
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = Config.model_validate(raw)
    # Relative paths are resolved against the config file's folder, not the CWD.
    if not cfg.data_dir.is_absolute():
        cfg.data_dir = (path.parent / cfg.data_dir).resolve()
    if not cfg.output_dir.is_absolute():
        cfg.output_dir = (path.parent / cfg.output_dir).resolve()
    return cfg
