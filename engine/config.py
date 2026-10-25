"""Paths and constants. Everything lives under one data folder the user owns."""
import json
import os
import sys
from pathlib import Path


def _default_home() -> Path:
    env = os.environ.get("AOME_HOME")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "AOMe"
    return Path.home() / ".aome"


HOME = _default_home()
DATASETS = HOME / "datasets"
JOBS = HOME / "jobs"
SETTINGS_FILE = HOME / "settings.json"   # tiny bootstrap file; always lives in HOME


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    cur = _load_settings()
    cur.update(d)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_FILE)


DEFAULT_MODELS = HOME / "models"


MODELS_FALLBACK = None             # set when the saved folder couldn't be opened (e.g. external drive unplugged)


def _resolve_models() -> Path:
    global MODELS_FALLBACK
    custom = _load_settings().get("models_dir")
    if custom:
        try:
            p = Path(custom)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            MODELS_FALLBACK = str(custom)   # fall back to the default folder, and tell the UI
    return DEFAULT_MODELS


# Downloaded base models (library). User-changeable: always read it as `config.MODELS` at call time.
MODELS = _resolve_models()


def set_models_dir(path) -> Path:
    """Point the model library at a new folder and remember it."""
    global MODELS, MODELS_FALLBACK
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    MODELS, MODELS_FALLBACK = p, None
    save_settings({"models_dir": str(p) if p.resolve() != DEFAULT_MODELS.resolve() else None})
    return p


AI_DIR = HOME / "ais"             # saved, finished AIs
EXPORTS = HOME / "exports"
LOGS = HOME / "logs"
DB_FILE = HOME / "aome.db"

for _p in (HOME, DATASETS, JOBS, MODELS, DEFAULT_MODELS, AI_DIR, EXPORTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 200
VERSION = "0.1.1"

# Chat models the training engine supports out of the box. Any Hugging Face
# causal LM the user downloads into the library can also be trained.
SUPPORTED_MODELS = [
    {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen 2.5 0.5B Instruct", "family": "chatbot",
     "parameters": "0.5B", "license": "Apache-2.0", "recommended_min_vram_mb": 3072},
    {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen 2.5 1.5B Instruct", "family": "chatbot",
     "parameters": "1.5B", "license": "Apache-2.0", "recommended_min_vram_mb": 6144},
    {"model_id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen 2.5 3B Instruct", "family": "chatbot",
     "parameters": "3B", "license": "Qwen-Research", "recommended_min_vram_mb": 10240},
    {"model_id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B Instruct", "family": "chatbot",
     "parameters": "7B", "license": "Apache-2.0", "recommended_min_vram_mb": 16384},
    {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "name": "Llama 3.2 1B Instruct", "family": "chatbot",
     "parameters": "1B", "license": "Llama 3.2", "recommended_min_vram_mb": 4096},
    {"model_id": "HuggingFaceTB/SmolLM2-360M-Instruct", "name": "SmolLM2 360M Instruct", "family": "chatbot",
     "parameters": "0.36B", "license": "Apache-2.0", "recommended_min_vram_mb": 2048},
]
