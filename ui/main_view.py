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
    with st.expander("Xem nguồn tài liệu được sử dụng"):
        if not sources:
            st.info("Không có nguồn nào để hiển thị.")
            return

        for i, source in enumerate(sources, start=1):
            st.markdown(f"**Nguồn {i}: {source['label']}**")
            st.caption(
                f"chunk={source.get('chunk_id')} | start={source.get('start_index')} | end={source.get('end_index')} | rerank={source.get('rerank_score')}"
            )
            st.info(source["content"])


def render_graph_rag():
    st.subheader("GraphRAG Integration")
    st.write("Tính năng GraphRAG cho phép truy vấn dựa trên đồ thị tri thức từ tài liệu.")
    st.info("Hiện tại tính năng này đang trong quá trình tích hợp. Vui lòng quay lại sau.")
    
    # Placeholder for GraphRAG UI
    st.text_input("Đặt câu hỏi (GraphRAG Mode)", key="graph_rag_question")
    if st.button("Truy vấn Graph"):
        st.warning("Công cụ GraphRAG chưa được cấu hình hoàn thiện.")
