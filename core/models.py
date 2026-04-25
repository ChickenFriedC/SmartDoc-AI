import streamlit as st
import torch

if not torch.cuda.is_available():
    torch.set_num_threads(4)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL, MODEL_NAME

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
def get_llm():
    return Ollama(
        model=MODEL_NAME,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        num_thread=8,
        num_ctx=4096,
        num_predict=512
    )
