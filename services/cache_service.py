import hashlib
import json
import os
import shutil

from config import CACHE_DIR


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def build_cache_key(file_hashes: list[str], chunk_size: int, chunk_overlap: int, embedding_model: str) -> str:
    raw_key = f"{'|'.join(file_hashes)}|{chunk_size}|{chunk_overlap}|{embedding_model}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_cache_dir(cache_key: str) -> str:
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, cache_key)


def cache_exists(cache_dir: str) -> bool:
    return (
        os.path.isdir(cache_dir)
        and os.path.exists(os.path.join(cache_dir, "index.faiss"))
        and os.path.exists(os.path.join(cache_dir, "index.pkl"))
        and os.path.exists(os.path.join(cache_dir, "meta.json"))
    )


def save_cache_metadata(cache_dir: str, metadata: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_cache_metadata(cache_dir: str) -> dict | None:
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_cache_dir() -> None:
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)