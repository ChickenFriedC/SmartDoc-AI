import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL, MODEL_NAME

from functools import lru_cache

@st.cache_resource(show_spinner=False)
def get_embedder(device: str = "cpu"):
    # Chuyển về CPU mặc định nếu không có GPU mạnh để ổn định luồng
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

@lru_cache(maxsize=1)
def _get_ollama_instance(model_name: str):
    # Dùng Singleton thực thụ qua lru_cache
    return Ollama(
        model=model_name,
        temperature=0.1,
        num_ctx=4096,
        stop=["<|im_end|>", "<|endoftext|>", "User:", "Assistant:"],
    )

def get_llm():
    return _get_ollama_instance(MODEL_NAME)