import streamlit as st
import torch
import os
import json
import urllib.request
import urllib.error
from typing import Any

if not torch.cuda.is_available():
    torch.set_num_threads(4)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL, MODEL_NAME


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _http_get_json(url: str, timeout_s: float = 2.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except Exception:
        return {"raw": data}


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_diagnose(model_name: str | None = None, base_url: str | None = None) -> dict[str, Any]:
    """Best-effort diagnosis for common Ollama runtime failures.

    Returns a dict like: { ok: bool, base_url, version?, model_present?, models?, error? }
    """
    base_url = (base_url or ollama_base_url()).rstrip("/")
    model_name = model_name or MODEL_NAME
    result: dict[str, Any] = {"ok": False, "base_url": base_url, "model": model_name}

    try:
        version = _http_get_json(f"{base_url}/api/version")
        result["version"] = version.get("version", version)
    except Exception as e:
        result["error"] = f"Cannot reach Ollama at {base_url}: {e}"
        return result

    try:
        tags = _http_get_json(f"{base_url}/api/tags")
        models = tags.get("models", []) if isinstance(tags, dict) else []
        result["models"] = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
        result["model_present"] = model_name in set(result["models"])
    except Exception as e:
        # Version is reachable; tags might fail on some setups.
        result["tags_error"] = str(e)

    result["ok"] = True
    return result


def is_ollama_runner_terminated_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return "llama runner process has terminated" in msg or "status code 500" in msg

@st.cache_resource(show_spinner=False)
def get_embedder(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

@st.cache_resource(show_spinner=False)
def get_llm(
    model: str | None = None,
    *,
    base_url: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    num_thread: int | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
):
    """Return a configured Ollama LLM.

    Values can be overridden via env vars:
    - OLLAMA_BASE_URL
    - OLLAMA_MODEL
    - OLLAMA_TEMPERATURE, OLLAMA_TOP_P, OLLAMA_REPEAT_PENALTY
    - OLLAMA_NUM_THREAD, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT
    """
    model = model or os.getenv("OLLAMA_MODEL", MODEL_NAME)
    base_url = base_url or ollama_base_url()

    temperature = _env_float("OLLAMA_TEMPERATURE", 0.7) if temperature is None else temperature
    top_p = _env_float("OLLAMA_TOP_P", 0.9) if top_p is None else top_p
    repeat_penalty = _env_float("OLLAMA_REPEAT_PENALTY", 1.1) if repeat_penalty is None else repeat_penalty
    num_thread = _env_int("OLLAMA_NUM_THREAD", 8) if num_thread is None else num_thread
    num_ctx = _env_int("OLLAMA_NUM_CTX", 4096) if num_ctx is None else num_ctx
    num_predict = _env_int("OLLAMA_NUM_PREDICT", 512) if num_predict is None else num_predict

    return Ollama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_thread=num_thread,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )


def get_llm_safe(model: str | None = None):
    """A lower-memory fallback configuration for when the Ollama runner crashes."""
    safe_ctx = _env_int("OLLAMA_SAFE_NUM_CTX", 2048)
    safe_predict = _env_int("OLLAMA_SAFE_NUM_PREDICT", 256)
    safe_threads = _env_int("OLLAMA_SAFE_NUM_THREAD", 4)
    return get_llm(
        model=model,
        num_ctx=safe_ctx,
        num_predict=safe_predict,
        num_thread=safe_threads,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1,
    )
