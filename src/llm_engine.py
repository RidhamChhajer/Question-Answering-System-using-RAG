"""
llm_engine.py
-------------
Thin wrapper around the Ollama Python SDK for local LLM inference.

Model: mistral
  - Runs fully locally via the Ollama daemon
  - No external API calls
  - Strong reasoning, good for RAG

Prerequisites:
  1. Install Ollama: https://ollama.com/download
  2. Pull the model once:  ollama pull mistral
  3. Install SDK:          pip install ollama>=0.3.0

Streaming is enabled so the user sees output token-by-token.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ollama

# The model tag to use — change to "llama3" or "phi3" etc. if preferred
DEFAULT_MODEL = config.OLLAMA_MODEL


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    stream: bool = True,
) -> str:
    """
    Send a prompt to the local Ollama LLM and return the generated response.

    Args:
        prompt: The fully assembled RAG prompt (context + question).
        model:  Ollama model tag.
        stream: If True, prints tokens as they arrive and returns full text.
                If False, waits for completion and returns silently.

    Returns:
        The full generated answer string.

    Raises:
        RuntimeError: If Ollama daemon is not running or model not pulled.
    """
    try:
        if stream:
            full_response = ""
            response_stream = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=config.OLLAMA_OPTIONS,
                stream=True,
            )
            for chunk in response_stream:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_response += token
            print()  # newline after streaming ends
            return full_response

        else:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=config.OLLAMA_OPTIONS,
            )
            return response["message"]["content"]

    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise RuntimeError(
                "❌ Cannot connect to Ollama.\n"
                "   Make sure Ollama is running: https://ollama.com/download\n"
                f"   Then pull the model: ollama pull {model}"
            ) from e
        if "not found" in error_msg or "does not exist" in error_msg:
            raise RuntimeError(
                f"❌ Model '{model}' not found in Ollama.\n"
                f"   Pull it first: ollama pull {model}"
            ) from e
        raise
