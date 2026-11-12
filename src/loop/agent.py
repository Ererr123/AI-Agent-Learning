"""
Local Agent for the local LLM model.

Setup:
    pip install openai
    llama-server -hf <your-gguf-repo>:Q4_K_M --jinja -c 16384 --port 8080
    python agent.py
"""

import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI


## Our planned Configuration for the local agent
BASE_URL = "http://localhost:8080/v1"  ## The base URL for the local LLM server
MODEL = "ChocoBanana"     ## The model name to use for the local LLM server
NOTES_DIR = Path("notes") ## the only folder the agent can read/write to
MAX_STEPS = 10             ## The maximum number of steps the agent can take
MAX_FILE_CHARS = 4000      ## Keep our AI small and fast by limiting the number of characters it can read from a file

SYSTEM_PROMPT = (
    "You are a personal assistant that can read through files in a local directory and "
    "write notes to them. You can use whatever tools you need to accomplish your goals, "
    "but you must always be careful to stay within the notes directory. "
    "Do not gueess or make up information. Keep your responses concise and to the point. "
)

client = OpenAI(base_url=BASE_URL)  ## Create an OpenAI client with the local LLM server