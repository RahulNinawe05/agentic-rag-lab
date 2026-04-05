from langgraph.graph import StateGraph, START, END
from src.state import GraphState
from src.nodes import (
    make_retriever_node,
    grade_node,
    web_search_node,
    generate_node,
    hallucination_check_node,
    decided_next_step,
    decide_after_hallucination
)

def build_graph(retriever):
    g = StateGraph(GraphState)

    g.add_node("retriever", make_retriever_node(retriever))
    g.add_node("grade", grade_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)
    g.add_node("hallucination_check", hallucination_check_node)

    g.add_edge(START, "retriever")
    g.add_edge("retriever", "grade")

    g.add_conditional_edges(
        'grade',
        decided_next_step,
        {"web_search": "web_search", "generate": "generate"}
    )

    g.add_edge("web_search", "generate")
    g.add_edge("generate", "hallucination_check")

    g.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination,
        {"web_search": "web_search", END: END}
    )

    return g.compile()