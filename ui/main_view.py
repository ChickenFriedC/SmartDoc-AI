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


def render_timing_summary(timings: dict, title: str = "⏱ Thời gian thực hiện"):
    if not timings:
        return

    st.subheader(title)

    total = 0.0
    for _, value in timings.items():
        total += value

    for step_name, seconds in timings.items():
        st.write(f"- **{step_name}**: `{seconds:.3f}s`")

    st.write(f"**Tổng thời gian**: `{total:.3f}s`")