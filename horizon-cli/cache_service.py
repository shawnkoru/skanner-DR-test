"""Simple filesystem caching for deep research outputs.

Stores per-topic artifacts:
  <slug>_dr.md
  <slug>_parsed.json

Slug is a lowercase, underscore-delimited version of the topic plus a short hash
to reduce collision risk when topics are long/similar.
"""

from pathlib import Path
import hashlib
import json
from typing import Optional, Tuple


def _slugify(topic: str) -> str:
    base = topic.strip().lower().replace(' ', '_')[:120]
    h = hashlib.sha256(topic.encode('utf-8')).hexdigest()[:8]
    return f"{base}_{h}"


def cache_paths(cache_dir: Path, topic: str) -> Tuple[Path, Path]:
    slug = _slugify(topic)
    return cache_dir / f"{slug}_dr.md", cache_dir / f"{slug}_parsed.json"


def load(cache_dir: Path, topic: str) -> Tuple[Optional[str], Optional[dict]]:
    dr_path, parsed_path = cache_paths(cache_dir, topic)
    dr_text = None
    parsed = None
    if dr_path.is_file():
        try:
            dr_text = dr_path.read_text(encoding='utf-8')
        except Exception:
            dr_text = None
    if parsed_path.is_file():
        try:
            parsed = json.loads(parsed_path.read_text(encoding='utf-8'))
        except Exception:
            parsed = None
    return dr_text, parsed


def save(cache_dir: Path, topic: str, dr_text: Optional[str], parsed: Optional[dict]):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dr_path, parsed_path = cache_paths(cache_dir, topic)
    if dr_text is not None:
        try:
            dr_path.write_text(dr_text, encoding='utf-8')
        except Exception:
            pass
    if parsed is not None:
        try:
            parsed_path.write_text(json.dumps(parsed, indent=2), encoding='utf-8')
        except Exception:
            pass
