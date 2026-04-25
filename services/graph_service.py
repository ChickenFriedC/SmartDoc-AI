import networkx as nx
import matplotlib.pyplot as plt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from core.models import get_llm

def build_knowledge_graph(documents):
    llm = get_llm()
    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(documents)
    G = nx.Graph()
    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            G.add_node(node.id, type=node.type)
        for edge in graph_doc.edges:
            G.add_edge(edge.source.id, edge.target.id, relation=edge.type)
    return G

def get_relevant_entities(G, question_entities):
    relevant_subgraph_nodes = set()
    for entity in question_entities:
        if G.has_node(entity):
            relevant_subgraph_nodes.add(entity)
            neighbors = list(G.neighbors(entity))
            relevant_subgraph_nodes.update(neighbors)
    return list(relevant_subgraph_nodes)

def visualize_graph(G):
    if len(G.nodes) == 0: return None
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_weight='bold', ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    return fig
