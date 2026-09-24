# podcast-digest

Python CLI: podcast link → transcript → Claude structured summary → Obsidian notes.
Code in `src/podcast_digest/`, tests in `tests/` (offline, fake Claude client).

## When the user pastes a podcast episode link (Spotify, Apple, RSS, audio URL)

That's a request to process it. Run from the repo root with the venv active:

```bash
source .venv/bin/activate && podcast-digest episode "<link>"
```

- Transcription can take several minutes for a 2-3 hour episode, so run it in the
  background and tell the user it has started.
- When it finishes, read the note path it prints and reply with the one-liner, the 5 summary
  bullets, and the "Relevant to" section for the first interest. Include the episode cost
  (`cost_usd` in the frontmatter).
- If resolving fails (e.g. a Spotify exclusive), say so and ask for the Apple Podcasts
  link, or the RSS feed plus a title keyword (`--match`).
- "Weekly digest" → `podcast-digest digest`. "What did it cost" → `podcast-digest costs`.

## Development

- `pytest && ruff check . && ruff format --check .` must pass.
- `config.yaml`, `data/`, and audio stay out of git. Secrets are env vars only
  (`ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`).
- Default model is `claude-sonnet-5` (the user chose it). Don't change it without asking.
