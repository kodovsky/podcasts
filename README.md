# podcast-digest

Paste a podcast link, get an Obsidian note with the substance:

**link** (Spotify, Apple Podcasts, RSS, audio file) → **transcript with timestamps** (the
feed's published transcript if it has one, else Whisper on your Mac's GPU, else Deepgram)
→ **Claude summary as structured JSON** → **Markdown note** in your vault, plus a
**weekly digest**.

Each episode note has: 5-bullet summary, non-obvious ideas, actionable advice (as
checkboxes with timestamps), books/tools mentioned (as `[[wikilinks]]`), ideas relevant to
each of your interests, and quotes. Sponsor reads are ignored.

## Setup (macOS, Apple Silicon)

```bash
brew install ffmpeg python@3.12           # mlx-whisper decodes audio with ffmpeg; needs Python 3.11+
git clone https://github.com/kodovsky/podcasts && cd podcasts
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[mac,dev]'               # use '.[local]' for faster-whisper instead
cp config.example.yaml config.yaml        # set output_dir to your vault + your interests
# Summaries run through Claude Code on your subscription (backend: claude-code in config.yaml):
# install Claude Code and run `claude` once to log in. For the pay-per-token API instead, set
# backend: api and export ANTHROPIC_API_KEY=...
export DEEPGRAM_API_KEY=...               # optional: fallback transcription
```

Everything stays in the project folder (both `notes/` and `data/` are git-ignored):

```
notes/                         point output_dir at your Obsidian vault when you're ready
  <Podcast>/<date> <title>.md
  <Podcast>/Transcripts/…
  Digests/
data/
  audio/<Podcast>/<date> <title> - <id>.mp3        (keep_audio: false deletes after transcription)
  transcripts/<Podcast>/<date> <title> - <id>.json
  models/                      Whisper model (~1.5 GB, downloaded on first use)
  state.sqlite, logs/
```

## First test: one downloaded MP3

```bash
# 1. Transcribe only (free, runs locally, no API key needed). Check the preview.
podcast-digest transcribe ~/Downloads/episode.mp3 --title "Episode title" --podcast "The Tim Ferriss Show"

# 2. Summarize + write the note (reuses the cached transcript; a few cents of Claude usage)
podcast-digest episode ~/Downloads/episode.mp3 --title "Episode title" --podcast "The Tim Ferriss Show"
podcast-digest costs
```

## Usage

```bash
podcast-digest episode "https://open.spotify.com/episode/…"          # Spotify link
podcast-digest episode "https://podcasts.apple.com/…/id863897795?i=…" # Apple link
podcast-digest episode https://rss.art19.com/tim-ferriss-show --match "Jane Doe"
podcast-digest episode ~/Downloads/episode.mp3
podcast-digest episode <link> --dry-run      # just show what the link resolves to

podcast-digest list ferriss                  # recent episodes of a configured feed
podcast-digest run                           # new episodes from all configured feeds
podcast-digest digest [--week 2026-W39]      # weekly digest of what you processed
podcast-digest costs                         # cost ledger (Claude tokens, Deepgram minutes)
```

Output layout in `output_dir`:

```
The Tim Ferriss Show/
  2026-09-18 #… Title.md
  Transcripts/2026-09-18 … Title (transcript).md
Digests/
  Podcast digest 2026-W39.md
```

## How it works

| Step | Details |
|---|---|
| Resolve | Spotify has no downloadable audio, so Spotify/Apple links are matched to the show's public RSS feed via the iTunes Search API (Spotify exclusives won't resolve). |
| Transcript | `<podcast:transcript>` from the feed if it's VTT/SRT/JSON (timed) → mlx-whisper (Apple GPU) / faster-whisper → Deepgram fallback. Cached in `data/transcripts/`, so a failed summary never re-transcribes. |
| Summarize | Claude Code on your subscription (`claude -p`, default) or the Claude API (`claude-sonnet-5`) with structured outputs validated by Pydantic. Transcripts over `max_chunk_tokens` are summarized per chunk (with overlap) and then merged. API errors are retried with backoff by the SDK; truncated/invalid output is retried too. |
| State | `data/state.sqlite`: processed episodes (so nothing is done twice) and a cost ledger. Logs in `data/logs/`. |

## Triggering from your phone

Run Claude Code on your Mac with Remote Control in this folder:

```bash
cd podcasts && source .venv/bin/activate && claude remote-control
```

Open that session in the Claude app on your phone and paste an episode link. `CLAUDE.md`
tells Claude to run the pipeline and reply with the summary.

## Development

```bash
pytest && ruff check . && ruff format --check .
```

Tests run offline, with fixtures and a fake Claude client.
