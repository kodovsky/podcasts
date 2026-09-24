# podcast-digest

Podcast RSS → local transcription (faster-whisper, Deepgram fallback) → Claude
structured summaries → Obsidian Markdown notes + weekly digest.

**Status:** scaffolding only; the pipeline is being designed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[local,dev]'          # add ,deepgram for the API fallback
cp config.example.yaml config.yaml     # then edit
export ANTHROPIC_API_KEY=...
podcast-digest run
```
