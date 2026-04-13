import streamlit as st

def render_sidebar():
    st.sidebar.title("⚙️ Cấu hình hệ thống")
    st.sidebar.markdown("---")
    st.sidebar.write("**Model:** Qwen2.5:7b")
    st.sidebar.write("**Embeddings:** Multilingual MPNet")
    st.sidebar.write("**Status:** Local Runtime (Ollama)")