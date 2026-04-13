import streamlit as st

def render_page_header():
    st.title("📄 SmartDoc AI - Intelligent Document Q&A System")
    st.write("Hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật RAG.")

def render_answer(response: str):
    st.subheader("💡 Trả lời:")
    st.write(response)

def render_sources(relevant_docs):
    with st.expander("🔍 Xem nguồn tài liệu được sử dụng"):
        for i, doc in enumerate(relevant_docs, start=1):
            st.markdown(f"**Đoạn trích {i}:**")
            st.info(doc.page_content)