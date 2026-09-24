"""CLI entry point (placeholder until the pipeline is implemented)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="podcast-digest")
    parser.add_argument("-c", "--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Fetch, transcribe and summarize new episodes")
    ep = sub.add_parser("episode", help="Process a single episode by URL")
    ep.add_argument("url")
    sub.add_parser("digest", help="Build the weekly digest")
    args = parser.parse_args()
    raise SystemExit(f"'{args.command}' is not implemented yet")


if __name__ == "__main__":
    main()
