from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_PATH = Path("models/research_review_model.json")
DEFAULT_CRITERIA_PATH = Path("data/Analyse Critaria/metadata.json")
DEFAULT_PAPERS_DIR = Path("data/Research Papers")


def load_env(path: Path = Path(".env")) -> None:
    """Small .env loader to avoid requiring python-dotenv at runtime."""
    if not path.exists():
        load_streamlit_secrets()
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    load_streamlit_secrets()


def load_streamlit_secrets() -> None:
    try:
        import streamlit as st
    except ImportError:
        return
    try:
        for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
            value = st.secrets.get(key)
            if value:
                os.environ.setdefault(key, str(value))
    except Exception:
        return


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
