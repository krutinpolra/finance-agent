"""
Phase 3 — a second agent: the Budget Advisor.

Same reason -> act -> observe loop as agent.py, just with a different system
prompt and a different tool. This is the core idea behind "multi-agent": each
agent is the same loop mechanism, specialized by what it's told to do and
what it's allowed to call. This agent's input is the categorizer agent's
output (a spending breakdown), not raw transactions — that handoff is what
makes this "multi-agent" rather than just a second prompt.
"""

# --- The tool's real implementation: plain Python, nothing AI about it ---
BENCHMARKS = {
    "Rent/Mortgage": 30,
    "Groceries": 10,
    "Dining & Restaurants": 5,
    "Coffee & Cafes": 2,
    "Transportation": 10,
    "Shopping": 5,
    "Subscriptions": 3,
    "Utilities": 8,
    "Entertainment": 5,
    "Health & Fitness": 5,
    "Other": 5,
}


def get_spending_benchmark(category: str) -> str:
    pct = BENCHMARKS.get(category)
    if pct is None:
        return "no benchmark available for this category"
    return f"typical recommended spending is around {pct}% of income"


# --- The schema: how we describe that tool to Claude ---
TOOLS = [
    {
        "name": "get_spending_benchmark",
        "description": (
            "Look up the typical recommended spending percentage of income "
            "for a category, based on common budgeting guidelines. Use this "
            "before judging whether a category's spend looks high or low."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "One of the transaction categories, e.g. 'Groceries'"
                }
            },
            "required": ["category"]
        }
    }
]

SYSTEM_PROMPT = """You are a personal budget advisor reviewing a spending
breakdown that another agent already categorized. You'll be given each
category's spend as a percentage of monthly income.

For each category, use the get_spending_benchmark tool to check it against
typical guidelines before judging it as high, low, or normal - don't guess.

Then write a short, plain-English summary for the person: 3-5 bullet points
covering where they're overspending, underspending, or on track, plus one
concrete, actionable suggestion. Be encouraging, not judgmental. Respond in
plain markdown text, not JSON - this is a report a person will read directly.
"""


def get_budget_advice(client, category_percentages: dict) -> str:
    """Runs the think -> act -> observe loop, reviewing a spending breakdown
    instead of raw transactions and returning written advice instead of
    structured data.
    """
    breakdown = "\n".join(
        f"- {category}: {pct:.1f}% of income"
        for category, pct in category_percentages.items()
    )
    messages = [{
        "role": "user",
        "content": f"Here is this month's spending breakdown by category:\n{breakdown}"
    }]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            # Claude can request several tool calls in one turn (parallel tool
            # use) - every tool_use block here needs a matching tool_result in
            # the next message, or the API rejects the follow-up request.
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": get_spending_benchmark(block.input["category"]),
                }
                for block in response.content if block.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})
            continue

        return next(b.text for b in response.content if b.type == "text")
