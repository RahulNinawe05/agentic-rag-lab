from src.loader import load_and_split
from src.vectorstore import get_vectorstore
from src.graph import build_graph

if __name__ == "__main__":
    pdf_path = "D:\\agentic-rag-lab\\attention.pdf"

    chunks    = load_and_split(pdf_path)
    vs        = get_vectorstore(chunks)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

    graph  = build_graph(retriever)
    result = graph.invoke({"question": "what is Scaled Dot-Product Attention & what We call our particular attention"})

    if result.get("web_search_needed"):
        print("*" * 50)
        print("Answer came from: WEB SEARCH")
    else:
        print("*" * 50)
        print("Answer came from: PDF")

    print(result["answer"])