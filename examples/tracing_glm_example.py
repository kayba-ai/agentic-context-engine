"""Example: instrument a GLM agent with Kayba tracing.

Run:
    uv run python examples/tracing_glm_example.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from ace.tracing import configure, start_span, trace

load_dotenv()

# --- Kayba tracing setup ---------------------------------------------------
configure(
    api_key=os.environ["KAYBA_SDK_KEY"],
    folder="examples",
)

# --- GLM client via Zhipu's OpenAI-compatible endpoint ----------------------
client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
MODEL = "glm-5.1"


# --- Traced helper functions ------------------------------------------------
@trace(name="llm_call", span_type="LLM")
def llm_call(messages: list[dict[str, str]]) -> str:
    """Send a chat completion to GLM and return the text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


@trace(name="research_agent")
def research_agent(topic: str) -> str:
    """Agent that gathers key facts about a topic."""
    with start_span("build_prompt") as span:
        messages = [
            {
                "role": "system",
                "content": "You are a research assistant. List 3 key facts.",
            },
            {"role": "user", "content": f"Research this topic: {topic}"},
        ]
        span.set_inputs({"topic": topic})
        span.set_outputs({"message_count": len(messages)})

    result = llm_call(messages)
    return result


@trace(name="summariser_agent")
def summariser_agent(facts: str) -> str:
    """Agent that summarises research into a single paragraph."""
    with start_span("build_prompt") as span:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summariser. Condense the following facts "
                    "into one concise paragraph."
                ),
            },
            {"role": "user", "content": facts},
        ]
        span.set_inputs({"facts_length": len(facts)})
        span.set_outputs({"message_count": len(messages)})

    result = llm_call(messages)
    return result


@trace(name="pipeline")
def run_pipeline(topic: str) -> str:
    """Two-agent pipeline: research → summarise."""
    facts = research_agent(topic)
    print(f"\n--- Research Agent ---\n{facts}")

    summary = summariser_agent(facts)
    print(f"\n--- Summariser Agent ---\n{summary}")

    return summary


# --- Main -------------------------------------------------------------------
if __name__ == "__main__":
    result = run_pipeline("The history of the Silk Road")
    print(f"\n--- Final result ---\n{result}")
