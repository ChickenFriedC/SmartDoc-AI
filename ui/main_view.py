import streamlit as st



def render_page_header():
    st.title("SmartDoc AI - Intelligent Document Q&A System")
    st.write("Hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật RAG.")



def render_answer(response: str, validation: dict | None = None):
    st.subheader("Trả lời")
    st.write(response)
    if validation:
        confidence = validation.get("confidence")
        supported = validation.get("supported")
        reason = validation.get("reason")
        st.caption(f"Self-RAG | supported={supported} | confidence={confidence} | reason={reason}")



def render_sources(sources: list[dict]):
    st.markdown("### Nguồn trích dẫn & Minh chứng")
    if not sources:
        st.info("Không có nguồn nào để hiển thị.")
        return

    for i, source in enumerate(sources, start=1):
        with st.expander(f"Nguồn {i}: {source['label']}"):
            st.markdown(f"**Nội dung trích xuất:**")
            st.markdown(f"> {source['content']}")
            
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"File: {source['filename']}")
            with cols[1]:
                if source.get('page'):
                    st.caption(f"Trang: {source['page']}")
                elif source.get('paragraph'):
                    st.caption(f"Đoạn: {source['paragraph']}")
