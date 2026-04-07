import os
import re
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import llm
from src.state import GraphState

# ── Retriever ──────────────────────────────────────────────
def make_retriever_node(retriever):
    def retriever_node(state: GraphState):
        docs = retriever.invoke(state['question'])
        if not docs:
            return {'context': []}
        return {
            'context': [
                f"(Page {d.metadata.get('page')}): {d.page_content}"
                for d in docs
            ]
        }
    return retriever_node

# ── Grader ─────────────────────────────────────────────────
GRADE_PROMPT = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
    You are a relevance checker. Your only job is to decide if the context can help answer the question.

    Rules:
    - Read the context carefully
    - Check if the context contains information that directly or partially answers the question
    - If the context is even slightly useful, say "yes"
    - If the context is completely off-topic or empty, say "no"
    - Do NOT answer the question yourself
    - Do NOT explain your reasoning
    - Reply with ONE word only: yes or no

    Context: {context}
    Question: {question}
    Answer:
    """
)

GRADE_CHAIN = GRADE_PROMPT | llm

def grade_node(state: GraphState):
    relevant_doc = []
    for doc in state['context']:
        result = GRADE_CHAIN.invoke({
            'context': doc,
            'question': state['question']
        })
        if 'yes' in result.content.lower():
            relevant_doc.append(doc)

    if not relevant_doc:
        return {'context': [], 'web_search_needed': True}
    return {'context': relevant_doc, 'web_search_needed': False}

# ── Web Search ─────────────────────────────────────────────
def web_search_node(state: GraphState):
    search = TavilySearchResults(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=3
    )
    result = search.invoke(state["question"])
    web_context = [r["content"] for r in result]
    return {'context': web_context}

# ── Router ─────────────────────────────────────────────────
def decided_next_step(state: GraphState):
    if state['web_search_needed']:
        return "web_search"
    return "generate"

# ── Generator ──────────────────────────────────────────────
PROMPT = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
    You are a helpful assistant.

    Rules:
    - Answer using ONLY the context provided
    - If context is missing any detail, still answer what you know
    - NEVER say "not found" or "insufficient" — always attempt an answer
    - Keep answer clear and specific

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    Answer:
    """
)
RAG_CHAIN = PROMPT | llm

def generate_node(state: GraphState):
    if not state['context']:
        return {"answer": "Not found in the document"}
    response = RAG_CHAIN.invoke({
        'context': "\n\n".join(state['context']),
        'question': state['question']
    })
    return {'answer': response.content}

# ── Hallucination Check ────────────────────────────────────
HALLUCINATION_PROMPT = PromptTemplate(
    input_variables=['context', 'answer'],
    template="""
    You are a strict fact-checker.

    Rules:
    - If ALL facts in the answer are clearly present in the context, say "yes"
    - If ANY part of the answer contains information NOT in the context, say "no"
    - If the answer says "not found" or "insufficient information", say "no"
    - Reply with ONE word only: yes or no

    Context: {context}
    Answer: {answer}
    Verdict:
    """
)
HALLUCINATION_CHAIN = HALLUCINATION_PROMPT | llm
MAX_RETRIES = 2

def hallucination_check_node(state: GraphState):
    current_retry = state.get('retry_count', 0)
    result = HALLUCINATION_CHAIN.invoke({
        "context": "\n\n".join(state['context']),
        "answer": state['answer']
    })
    is_hallucination = bool(re.search(r'\bno\b', result.content.lower()))

    if is_hallucination and current_retry < MAX_RETRIES:
        print(f"Hallucination detected! Routing to web search... (Attempt {current_retry + 1}/{MAX_RETRIES})")
        return {
            "hallucination_detected": True,
            "web_search_needed": True,
            "retry_count": current_retry + 1
        }
    if is_hallucination and current_retry >= MAX_RETRIES:
        print(f"Max retries ({MAX_RETRIES}) reached! Returning best available answer.")
        return {
            "hallucination_detected": False,
            "retry_count": current_retry
        }

    print("Answer is grounded in context.")
    return {
        "hallucination_detected": False,
        "retry_count": current_retry
    }

# ── Post Hallucination Router ──────────────────────────────
def decide_after_hallucination(state: GraphState):
    from langgraph.graph import END
    if state['hallucination_detected']:
        return "web_search"
    return END