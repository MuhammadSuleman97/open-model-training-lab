#!/usr/bin/env python3
"""Publish only the static learning guide to a public Hugging Face Space."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_NAME = "open-model-training-lab"
UPLOADS = {
    "README.md": PROJECT_ROOT / "docs" / "HUGGING_FACE_SPACE_CARD.md",
    "index.html": PROJECT_ROOT / "learning" / "index.html",
    "styles.css": PROJECT_ROOT / "learning" / "styles.css",
    "app.js": PROJECT_ROOT / "learning" / "app.js",
}
FORBIDDEN_TEXT = {
    "absolute macOS path": re.compile("/" + r"Users/[^/\s]+/"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
}


def validate_uploads() -> None:
    errors: list[str] = []
    for destination, source in UPLOADS.items():
        if not source.is_file():
            errors.append(f"missing {destination}: {source}")
            continue
        text = source.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_TEXT.items():
            if pattern.search(text):
                errors.append(f"{label} found in {source.relative_to(PROJECT_ROOT)}")
    if errors:
        raise SystemExit("\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-name", default=DEFAULT_SPACE_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_uploads()

    api = HfApi()
    account = api.whoami()
    owner = account["name"]
    repo_id = f"{owner}/{args.space_name}"

    print("Open Model Training Lab — Hugging Face Space publication")
    print(f"authenticated_owner: {owner}")
    print(f"space_id: {repo_id}")
    print("visibility: public")
    print("sdk: static")
    print(f"upload_files: {', '.join(UPLOADS)}")
    print("model_weights_uploaded: False")
    print("dataset_rows_uploaded: False")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        commit_message="Publish interactive Open Model Training Lab",
        operations=[
            CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
            for destination, source in UPLOADS.items()
        ],
    )

    print(f"commit_url: {commit.commit_url}")
    print(f"space_url: https://huggingface.co/spaces/{repo_id}")
    print("huggingface_space_publication_ok: True")


if __name__ == "__main__":
    main()
