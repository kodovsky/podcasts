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
    assert list((tmp_path / "data" / "transcripts").glob("*.json"))

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
