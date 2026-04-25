import os
import threading
from datetime import datetime

def _run_export(file_path: str, metrics: dict, history_metrics: list = None):
    if history_metrics is None:
        history_metrics = []
        
    total_queries = len(history_metrics)
    if total_queries > 0:
        supported_count = sum(1 for m in history_metrics if m.get('supported') is True or m.get('supported') == 'True')
        avg_confidence = sum(float(m.get('confidence', 0) or 0) for m in history_metrics) / total_queries
        high_accuracy_count = sum(1 for m in history_metrics if float(m.get('confidence', 0) or 0) > 0.8)
        
        actual_retrieval = f"{round((supported_count / total_queries) * 100, 1)}%"
        actual_relevance = f"{round(avg_confidence * 100, 1)}%"
        actual_accuracy = f"{round((high_accuracy_count / total_queries) * 100, 1)}%"
    else:
        actual_retrieval = "N/A (No queries yet)"
        actual_relevance = "N/A"
        actual_accuracy = "N/A"

    report_content = f"""==================================================
SMARTDOC AI - ACTUAL PERFORMANCE REPORT
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

[Configuration Used]
- Chunk Size:           {metrics.get('chunk_size', 'N/A')}
- Chunk Overlap:        {metrics.get('chunk_overlap', 'N/A')}
- Retriever Mode:       {metrics.get('retriever_mode', 'N/A')}

[Actual Processing Time - Latest Operation]
- Document Loading:     {metrics.get('load_time', 'N/A')} seconds (Target: 2-5s)
- Embedding Generation: {metrics.get('embed_time', 'N/A')} seconds (Target: 5-10s/100 chunks)
- Query Processing:     {metrics.get('query_time', 'N/A')} seconds (Target: 1-3s)
- Answer Generation:    {metrics.get('gen_time', 'N/A')} seconds (Target: 3-8s)

[Actual Accuracy Metrics - Calculated from {total_queries} queries]
- Relevant Retrieval:   {actual_retrieval} (Target: 85-90%)
- Answer Relevance:     {actual_relevance} (Target: 80-85%)
- Factual Accuracy:     {actual_accuracy} (Target: 75-80%)

==================================================
System Info:
- Model: Qwen2.5:7b
- Embeddings: Multilingual MPNet
- Accuracy Calculation: Dynamic based on Self-RAG logs
==================================================
"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except:
        pass

def export_performance_report_async(metrics: dict, history_metrics: list = None, file_path: str = "performance_report.txt"):
    """
    Triggers the report export in a background thread with real-time dynamic calculation.
    """
    thread = threading.Thread(target=_run_export, args=(file_path, metrics, history_metrics), daemon=True)
    thread.start()
