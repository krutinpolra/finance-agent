"""
Phase 1 — the actual agent.

This is a hand-rolled tool-calling loop, no framework. The point of writing
it yourself once is that every multi-agent framework you use later (LangGraph,
CrewAI, the Claude Agent SDK) is automating exactly this mechanism.
"""

import json
import re

# --- The tool's real implementation: plain Python, nothing AI about it ---
KNOWN_MERCHANTS = {
    "starbucks": "Coffee & Cafes",
    "netflix": "Subscriptions",
    "uber": "Transportation",
    "whole foods": "Groceries",
    "shell": "Gas & Fuel",
    "spotify": "Subscriptions",
    "amazon": "Shopping",
    "amzn": "Shopping",
}


def lookup_known_merchant(merchant_keyword: str) -> str:
    keyword = merchant_keyword.lower()
    for key, category in KNOWN_MERCHANTS.items():
        if key in keyword or keyword in key:
            return category
    return "not found in local lookup table"


# --- The schema: how we describe that tool to Claude ---
TOOLS = [
    {
        "name": "lookup_known_merchant",
        "description": (
            "Check a local table for a known category mapping for a merchant. "
            "Always try this first - it's instant and free. If it returns "
            "'not found', use your own judgment instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "merchant_keyword": {
                    "type": "string",
                    "description": "A short keyword from the transaction, e.g. 'starbucks'"
                }
            },
            "required": ["merchant_keyword"]
        }
    }
]

SYSTEM_PROMPT = """You are a personal finance categorization assistant.
For each transaction, first try the lookup_known_merchant tool. If it
doesn't find a match, use your own judgment to pick the single best
category from this list: Groceries, Dining & Restaurants, Coffee & Cafes,
Transportation, Shopping, Subscriptions, Utilities, Rent/Mortgage,
Entertainment, Health & Fitness, Income, Other.

When you have a final answer, respond with ONLY a JSON object on one line,
nothing else: {"category": "...", "confidence": "high" or "low"}
"""


def categorize_transaction(client, description: str, amount: float) -> dict:
    """Runs the think -> act -> observe loop for a single transaction.

    Returns a dict like {"category": "Coffee & Cafes", "confidence": "high"}.
    """
    messages = [{
        "role": "user",
        "content": f"Transaction: '{description}', amount: ${amount}."
    }]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Add Claude's turn to the conversation, whatever it contained
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_call = next(b for b in response.content if b.type == "tool_use")
            result = lookup_known_merchant(tool_call.input["merchant_keyword"])

            # Hand the result back, tagged with which tool call it answers
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                }]
            })
            continue  # loop back - Claude sees the result and responds again

        # stop_reason wasn't "tool_use", so this is Claude's final answer
        final_text = next(b.text for b in response.content if b.type == "text")

        # Claude doesn't always return *pure* JSON - it can wrap the object in a
        # markdown code fence or add a stray sentence. Pull out just the {...}
        # substring instead of requiring the whole string to be valid JSON.
        match = re.search(r"\{.*\}", final_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"[agent] Could not parse JSON from Claude's response: {final_text!r}")
        return {"category": "Uncategorized", "confidence": "low"}
