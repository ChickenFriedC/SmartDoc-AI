import os
import sys
import time
from langchain_core.documents import Document

# Thêm thư mục gốc vào path để import được các module core/services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import build_vector_store
from services.qa_service import answer_question
from services.retrieval_service import build_hybrid_retriever

def run_test_case(name, doc_path, question, expected_desc, keywords):
    print(f"\n--- Running {name} ---")
    print(f"Question: {question}")
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    docs = [Document(page_content=content, metadata={"filename": os.path.basename(doc_path)})]
    
    vector_store = build_vector_store(docs, device="cpu")
    retriever = build_hybrid_retriever(vector_store, docs)
    
    result = answer_question(
        retriever, 
        question, 
        chat_history=[], 
        query_rewrite=False, 
        self_rag=True
    )
    
    from core.models import get_llm
    llm = get_llm()
    answer = llm.invoke(result["prompt"])
    
    print(f"Result: {answer}")
    
    # Tự động kiểm tra từ khóa (Tăng tính minh bạch)
    is_passed = any(kw.lower() in answer.lower() for kw in keywords)
    
    if is_passed:
        print(f"Status: ✓ PASSED")
    else:
        print(f"Status: ✗ FAILED (Keywords {keywords} not found in response)")

if __name__ == "__main__":
    print("SMARTDOC AI - AUTOMATED TEST SUITE")
    
    # Test Case 1: Simple Factual Question
    run_test_case(
        "Test Case 1: Simple Factual Question",
        "tests/data/manual.txt",
        "What is the installation procedure?",
        "Step-by-step instructions",
        ["Unpack", "cable", "button", "Connect"]
    )
    
    # Test Case 2: Complex Reasoning
    run_test_case(
        "Test Case 2: Complex Reasoning",
        "tests/data/research.txt",
        "What are the main findings and their implications?",
        "Summary with analysis",
        ["20%", "Findings", "training", "productivity"]
    )
    
    # Test Case 3: Out-of-context Question
    run_test_case(
        "Test Case 3: Out-of-context Question",
        "tests/data/recipe.txt",
        "How to solve differential equations?",
        "'I don't know' response",
        ["don't know", "know", "không biết", "chưa rõ"]
    )
