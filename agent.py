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

# Used for the Phase 2 self-correction pass: a transaction only reaches this
# prompt after the first pass already tried the tool and its own judgment and
# still came back "low" - so this version gets category definitions to work
# from and is pushed to commit to an answer instead of defaulting to "low" again.
SYSTEM_PROMPT_RECHECK = """You are re-checking a bank transaction that was categorized
with low confidence on a first pass. Look more closely and give a more careful
final answer. The lookup_known_merchant tool is still available if you haven't
ruled it out yet.

Category definitions:
- Groceries: supermarkets, grocery stores
- Dining & Restaurants: restaurants, fast food, takeout, delivery
- Coffee & Cafes: coffee shops, cafes
- Transportation: rideshare, public transit, parking, tolls, fuel
- Shopping: retail, online marketplaces, general merchandise
- Subscriptions: recurring digital services (streaming, software, memberships)
- Utilities: electricity, water, gas, internet, phone bills
- Rent/Mortgage: housing payments
- Entertainment: movies, events, hobbies, recreation
- Health & Fitness: gyms, medical, pharmacies
- Income: deposits, paychecks, refunds
- Other: anything that doesn't clearly fit above

First, reason in one short sentence about what the merchant or description most
likely is. Then commit to your best answer - only use "low" confidence if the
description is genuinely ambiguous even after reasoning about it (e.g. a generic
transfer with no merchant info at all).

Respond with ONLY a JSON object on one line, nothing else:
{"category": "...", "confidence": "high" or "low", "reasoning": "..."}
"""


def categorize_transaction(client, description: str, amount: float, strict: bool = False) -> dict:
    """Runs the think -> act -> observe loop for a single transaction.

    With strict=True, uses the stricter re-check prompt (Phase 2 self-correction
    pass) instead of the normal first-pass prompt.

    Returns a dict like {"category": "Coffee & Cafes", "confidence": "high"}.
    """
    system_prompt = SYSTEM_PROMPT_RECHECK if strict else SYSTEM_PROMPT
    messages = [{
        "role": "user",
        "content": f"Transaction: '{description}', amount: ${amount}."
    }]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Add Claude's turn to the conversation, whatever it contained
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            # Claude can request several tool calls in one turn (parallel tool
            # use) - every tool_use block here needs a matching tool_result in
            # the next message, or the API rejects the follow-up request.
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": lookup_known_merchant(block.input["merchant_keyword"]),
                }
                for block in response.content if block.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})
            continue  # loop back - Claude sees the result(s) and responds again

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
