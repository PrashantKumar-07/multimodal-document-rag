from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Sample manifest must contain a JSON list.")
    return value


def download_sample(sample: dict[str, Any], cache_dir: Path) -> tuple[str, bytes]:
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / str(sample["filename"])
    if path.exists():
        content = path.read_bytes()
    else:
        response = requests.get(str(sample["url"]), timeout=60)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ValueError(f"The sample URL for {sample['label']} did not return a PDF.")
        path.write_bytes(content)
    expected = str(sample.get("sha256", "")).strip().lower()
    if expected and hashlib.sha256(content).hexdigest() != expected:
        path.unlink(missing_ok=True)
        raise ValueError(f"Checksum validation failed for {sample['label']}.")
    return str(sample["filename"]), content
