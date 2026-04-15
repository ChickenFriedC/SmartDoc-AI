import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL, MODEL_NAME

@st.cache_resource(show_spinner=False)
def get_embedder(device: str = "cuda"):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    return Ollama(
        model=MODEL_NAME,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
    )