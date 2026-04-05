from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    web_search_needed: bool
    hallucination_detected: bool
    retry_count: int