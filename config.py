PAGE_TITLE = "SmartDoc AI+"
LAYOUT = "wide"

MODEL_NAME = "qwen2.5:7b"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CACHE_DIR = "src/data/cache"
MAX_FILE_SIZE_MB = 200

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

RETRIEVER_K = 3
RETRIEVER_FETCH_K = 20