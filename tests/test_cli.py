from podcast_digest import cli, transcribe
from podcast_digest.models import Segment, Transcript


def test_transcribe_local_file(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")
    config = tmp_path / "config.yaml"
    config.write_text(f"output_dir: {tmp_path / 'vault'}\ndata_dir: data\n")

    calls = []

    def fake_engine(engine, ep, cfg):
        calls.append(engine)
        return Transcript(
            [Segment(0, 5, "Hello and welcome.", None)], source="mlx-whisper", model="fake"
        )

    monkeypatch.setattr(transcribe, "_run_engine", fake_engine)
    monkeypatch.setattr(transcribe, "auto_engine", lambda: "mlx-whisper")
    args = ["-c", str(config), "transcribe", str(audio), "--title", "My Ep", "--podcast", "Show"]

    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "Show — My Ep" in out and "[00:00] Hello and welcome." in out
    assert list((tmp_path / "data" / "transcripts" / "Show").glob("* My Ep - *.json"))

    # Second run hits the cache: no new transcription.
    assert cli.main(args) == 0
    assert calls == ["mlx-whisper"]


def test_everything_lives_in_project_folder(tmp_path, monkeypatch):
    from podcast_digest.config import load_config

    project = tmp_path / "podcasts"
    project.mkdir()
    (project / "config.yaml").write_text("interests: [x]\n")
    monkeypatch.chdir(tmp_path)  # paths must resolve against the config, not the CWD
    cfg = load_config(project / "config.yaml")
    assert cfg.output_dir == project / "notes"
    assert cfg.models_dir == project / "data" / "models"

    monkeypatch.setenv("HF_HOME", "/elsewhere")
    transcribe._use_project_model_cache(cfg)
    import os

    assert os.environ["HF_HOME"] == str(project / "data" / "models")


def test_storage_layout_and_audio_cleanup(tmp_path, monkeypatch, db, cfg):
    from datetime import UTC, datetime

    from podcast_digest.models import Episode

    ep = Episode(
        title="Noam Brown: Agents",
        podcast="Dwarkesh Podcast",
        audio_url="https://cdn/x.mp3",
        published=datetime(2026, 9, 17, tzinfo=UTC),
    )
    # Files from the old flat layout get moved into <Podcast>/<date> <title> - <id>.*
    cfg.audio_dir.mkdir(parents=True)
    (cfg.audio_dir / f"{ep.id}.mp3").write_bytes(b"audio")
    cfg.transcripts_dir.mkdir(parents=True)
    old_t = Transcript([Segment(0, 1, "hi")], source="mlx-whisper")
    import json

    (cfg.transcripts_dir / f"{ep.id}.json").write_text(json.dumps(old_t.to_dict()))

    stem = f"2026-09-17 Noam Brown Agents - {ep.id}"
    assert transcribe.transcript_path(ep, cfg) == (
        cfg.transcripts_dir / "Dwarkesh Podcast" / f"{stem}.json"
    )
    assert transcribe._audio_path(ep, cfg) == cfg.audio_dir / "Dwarkesh Podcast" / f"{stem}.mp3"
    assert not (cfg.audio_dir / f"{ep.id}.mp3").exists()

    # keep_audio: false deletes the downloaded audio once transcribed
    cfg.transcription.keep_audio = False
    transcribe._cleanup_audio(ep, cfg)
    assert not list(cfg.audio_dir.rglob("*.mp3"))
