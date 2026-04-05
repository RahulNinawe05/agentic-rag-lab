import os
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not os.getenv('GROQ_API_KEY'):
    raise ValueError("GROQ_API_KEY not set")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0,
)