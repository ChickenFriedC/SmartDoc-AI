import os
import pickle
import faiss

from config import CACHE_DIR
from core.utils import get_file_hash

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_paths(temp_file_path: str):
    ensure_cache_dir()
    file_hash = get_file_hash(temp_file_path)
    faiss_path = os.path.join(CACHE_DIR, f"{file_hash}.faiss")
    vector_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")
    return faiss_path, vector_path

def cache_exists(faiss_path: str, vector_path: str) -> bool:
    return os.path.exists(faiss_path) and os.path.exists(vector_path)

def save_cache(vector_store, faiss_path: str, vector_path: str):
    faiss.write_index(vector_store.index, faiss_path)
    with open(vector_path, "wb") as f:
        pickle.dump(vector_store, f)

def load_cache(faiss_path: str, vector_path: str):
    _ = faiss.read_index(faiss_path)
    with open(vector_path, "rb") as f:
        vector_store = pickle.load(f)
    return vector_store