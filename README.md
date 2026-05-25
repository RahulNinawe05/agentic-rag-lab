# Agentic RAG Chatbot

A chatbot that reads PDF files and answers questions from them. 
Built using LangChain, LangGraph and Groq LLM.

## What does this project do?

I built this project to understand how RAG (Retrieval-Augmented Generation) 
works in an agentic way. You give it a PDF, it reads and stores the content, 
and when you ask a question it finds the most relevant parts and gives you an answer.

If the PDF doesn't have the answer, it automatically searches the web using Tavily.
It also checks if the answer is hallucinated or not before showing it to you.

## Features

- Reads and chunks PDF documents
- Stores content in ChromaDB (local vector store)
- Uses LangGraph to manage the workflow step by step
- Falls back to web search if the document doesn't have the answer
- Uses Groq's Llama 3.1 model for fast responses
- Hallucination check before giving final answer

## Project Structure

```
AGENTIC-RAG/
│
├── src/
│   ├── config.py        - all config stuff like LLM setup
│   ├── graph.py         - the main langgraph workflow
│   ├── loader.py        - loads and splits the PDF
│   ├── nodes.py         - each step of the agent (retrieve, grade, generate etc)
│   ├── state.py         - shared state between nodes
│   └── vectorstore.py   - sets up chromadb
│
├── chroma_db/           - vector store gets saved here automatically
├── .env                 - my api keys (not pushed to github)
├── .gitignore
├── attention.pdf        - the pdf i used for testing
├── main.py              - run this to start
└── requirements.txt
```


## How it works

```
PDF Input
    ↓
Split into Chunks
    ↓
Store in ChromaDB
    ↓
User Asks a Question
    ↓
Retrieve Relevant Chunks
    ↓
Grade if Chunks are Useful
    ↓              ↓
Not Useful       Useful
    ↓              ↓
Web Search    Generate Answer
    ↓              ↓
    └──── Check for Hallucination ────┘
                   ↓
            Final Answer 
```

## Setup

**Requirements:**
- Python 3.10 or above
- Groq API key (free to get)
- Tavily API key (optional, only needed for web search)

**Steps:**

```bash
# clone the repo
git clone https://github.com/RahulNinawe05/agentic-rag-lab
cd agentic-rag

# create virtual environment
python -m venv .myvenvr
.myvenvr\Scripts\activate    # windows

# install dependencies
pip install -r requirements.txt
```

Create a `.env` file and add keys:
```bash
GROQ_API_KEY=your_key_here        # Get it from: https://console.groq.com/keys
TAVILY_API_KEY=your_key_here      # Get it from: https://app.tavily.com/
``` 


## Run

```bash
python main.py
```

It will load the PDF, build the vector store and answer the question 
defined in main.py. You can change the question directly in that file.

## Tech used

- LangChain
- LangGraph
- Groq (Llama 3.1 8B)
- ChromaDB
- HuggingFace Embeddings
- Tavily Search
- Python dotenv