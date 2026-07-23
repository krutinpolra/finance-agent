# Project context — Personal Finance Agent

This is a handoff doc summarizing everything decided and built so far, so work can
continue in VS Code (or with any other assistant/tool) without losing context.

## The goal

Learn AI agents, "loop engineering" (the reason → act → observe agent loop, plus a
self-correction loop), and multi-agent systems — by building one real project over
about a month, from a completely from-scratch level. Full-stack developer, new to AI
agents specifically. End goal: a portfolio-ready project, not just a tutorial follow-along.

## Working style agreed on

Acting as mentor/guide/project manager for this build:
- Make sensible default decisions (stack, scope) rather than asking about every choice,
  and explain the reasoning.
- Move in small phases with a clear milestone and checkpoint at the end of each.
- Give real, runnable code — but explain every new concept inline as it's introduced,
  not just hand over finished files with no explanation.
- Prioritize free / near-free tools throughout (learning project, not production).

## The chosen project: Personal Finance Agent

Reads a CSV of bank/credit-card transactions, categorizes each one using an AI agent
with tool-calling, and (coming up) self-corrects anything it's unsure about, then
gives a spending summary. Chosen because it's a real problem, and it naturally covers
both halves of the learning goal: the core agent loop now, multi-agent later.

## Stack decisions (and why)

- **Python + Streamlit** — one file gives UI and backend together, so early weeks can
  focus on agent logic instead of frontend/backend plumbing. (Full-stack background
  means upgrading to a real API + separate frontend later, if wanted, will be fast.)
- **Anthropic Claude API**, `claude-haiku-4-5` model — cheapest current model, priced
  at $1 / $5 per million input/output tokens, plenty for this project's scale.
- **GitHub**, public repo (required for the free hosting option below, and better for
  a portfolio anyway).
- **Streamlit Community Cloud** for hosting — free, deploys straight from the GitHub
  repo, auto-redeploys on push. Trade-off: free apps sleep after inactivity and take
  ~30-50s to wake up on the next visit. Fine for a portfolio demo.
- **No database yet** — CSV upload in, results shown in-session. SQLite only gets
  added later if persistence across visits is actually needed.

## Roadmap / status

- [x] **Phase 0** — repo scaffold, `.env` handling, a "Test Claude connection" button
      to prove the full pipeline (Streamlit → API key → Claude → response) works.
- [x] **Phase 1** — hand-rolled agent loop with real tool-calling: upload a CSV, each
      transaction gets categorized. *(Just finished — details below.)*
- [ ] **Phase 2** — self-correction loop: anything the agent marks `"confidence": "low"`
      gets automatically re-checked with a closer look before being shown.
- [ ] **Phase 3** — multi-agent: add a Budget Advisor agent that reviews the
      categorized data and gives recommendations (planner/critic-style handoff).
- [ ] **Phase 4** — polish for portfolio: a small eval set (% categorized correctly),
      cost/latency logging, deployed live demo link, short write-up explaining the
      loop design decisions.

## Issue hit and resolved

Got `anthropic.BadRequestError: ... credit balance is too low`. This was **not a code
bug** — the traceback showed the request reached Anthropic's servers fine. The Claude
API is pay-as-you-go with no free tier (separate from claude.ai's free chat tier); fix
was adding a card + a few dollars of prepaid credit at console.anthropic.com (also
reachable at platform.claude.com) → Settings → Billing.

## Key concepts covered so far

- **The agent loop**: call the model → check `stop_reason` → if `"tool_use"`, run the
  requested tool and feed the result back in → repeat until the model gives a final
  answer instead of a tool request. There's no fixed number of steps; it loops until
  the model itself is done.
- **Tools are just a schema you hand the model** — a `name`, a plain-English
  `description` of what it does and when to use it, and an `input_schema` for its
  arguments. The model can't run your code directly; it requests a tool call, your
  code actually executes it, and you hand the result back.
- **Messages are the model's only memory.** LLMs don't remember previous API calls —
  the whole conversation (including the model's own past turns and tool calls) has to
  be resent every time. `messages.append(...)` after every response is how "memory"
  actually works here.
- **`tool_result` messages** go in with `role: "user"`, tagged with `tool_use_id` so
  the model knows which tool call the result answers.
- **Deterministic tool vs. model judgment**: give the agent a cheap, instant, reliable
  lookup (a plain Python dict, in this case known merchants → categories) for anything
  you already know the answer to, and only spend the model's reasoning on genuinely
  ambiguous cases. This is a real design pattern, not just a teaching device.
- **Ask for structured output** (a single-line JSON reply) so the response can be
  parsed programmatically instead of extracting an answer from free-form prose —
  with a `try/except` fallback, since models don't always follow the format perfectly.

## Current file contents

### `README.md`
```markdown
# Personal Finance Agent

An AI agent that reads your bank/credit-card transactions, categorizes them, catches its own
mistakes, and gives you a plain-English spending summary — built from scratch to learn agent
loops and multi-agent systems.

## Status
🚧 Phase 1 — hand-rolled agent loop with tool-calling categorizes uploaded transactions.

## Roadmap
- [x] Phase 0 — repo, environment, "hello agent" connectivity test
- [x] Phase 1 — raw agent loop: categorize a CSV of transactions, tool-calling by hand
- [ ] Phase 2 — self-correction loop: agent flags low-confidence categories and re-checks itself
- [ ] Phase 3 — multi-agent: add a Budget Advisor agent that reviews the categorized data
- [ ] Phase 4 — polish: eval set, cost logging, deployed demo, write-up

## Stack
- Python + [Streamlit](https://streamlit.io) (UI + backend in one)
- [Anthropic Claude API](https://docs.claude.com) (Haiku model)
- Hosted free on [Streamlit Community Cloud](https://streamlit.io/cloud)

## Running locally
\`\`\`bash
python -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Anthropic API key into .env
streamlit run app.py
\`\`\`

## Deployment
Deployed automatically from the `main` branch via Streamlit Community Cloud.
Live demo: _(add link here once deployed)_
```

### `requirements.txt`
```
streamlit>=1.38
anthropic>=0.40
pandas>=2.2
python-dotenv>=1.0
```

### `.gitignore`
```
venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
.DS_Store
*.csv
!sample_data/*.csv
```

### `.env.example`
```
ANTHROPIC_API_KEY=your-key-here
```

### `app.py`
```python
"""
Personal Finance Agent — Phase 1.

Step 1 is the same connectivity check from Phase 0. Step 2 now runs the real
agent loop (see agent.py) over every row of an uploaded CSV.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from agent import categorize_transaction

load_dotenv()  # loads ANTHROPIC_API_KEY from .env when running locally

st.set_page_config(page_title="Personal Finance Agent", page_icon="💵")
st.title("Personal Finance Agent")
st.caption("Phase 1 — agent loop with tool-calling, categorizing real transactions.")


def get_client() -> Anthropic:
    # Locally: reads from .env. On Streamlit Community Cloud: reads from
    # the "Secrets" you'll set in the app dashboard (same variable name).
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("No ANTHROPIC_API_KEY found. Add it to your .env file (local) "
                 "or Streamlit secrets (deployed).")
        st.stop()
    return Anthropic(api_key=api_key)


st.subheader("Step 1: confirm the pipeline works")
if st.button("Test Claude connection"):
    client = get_client()
    with st.spinner("Calling Claude..."):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": "In one short sentence, confirm you're ready to help categorize bank transactions."
            }]
        )
    st.success(response.content[0].text)

st.divider()
st.subheader("Step 2: categorize your transactions")
st.caption("No file? Use the sample at sample_data/transactions.csv to try it out.")

uploaded = st.file_uploader("Upload a transactions CSV (date, description, amount)", type="csv")

if uploaded and st.button("Categorize transactions"):
    client = get_client()
    df = pd.read_csv(uploaded)

    categories = []
    confidences = []
    progress = st.progress(0, text="Starting...")

    for i, row in df.iterrows():
        result = categorize_transaction(client, row["description"], row["amount"])
        categories.append(result["category"])
        confidences.append(result["confidence"])
        progress.progress(
            (i + 1) / len(df),
            text=f"Categorized {i + 1}/{len(df)}: {row['description'][:30]}"
        )

    df["category"] = categories
    df["confidence"] = confidences
    progress.empty()

    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.groupby("category")["amount"].sum().abs())
```

### `agent.py`
```python
"""
Phase 1 — the actual agent.

This is a hand-rolled tool-calling loop, no framework. The point of writing
it yourself once is that every multi-agent framework you use later (LangGraph,
CrewAI, the Claude Agent SDK) is automating exactly this mechanism.
"""

import json

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
        try:
            return json.loads(final_text)
        except json.JSONDecodeError:
            return {"category": "Uncategorized", "confidence": "low"}
```

### `sample_data/transactions.csv`
```csv
date,description,amount
2026-07-01,STARBUCKS STORE #4521,-6.75
2026-07-01,NETFLIX.COM,-15.49
2026-07-02,UBER TRIP HELP.UBER.COM,-23.10
2026-07-02,WHOLE FOODS MARKET,-84.32
2026-07-03,SHELL OIL 57443210,-42.00
2026-07-03,PAYCHECK DIRECT DEP,2400.00
2026-07-04,RIVERSIDE YOGA STUDIO,-18.00
2026-07-05,LOCAL HARDWARE CO,-27.65
2026-07-06,SPOTIFY PREMIUM,-11.99
2026-07-07,GREENLEAF DINER,-32.40
2026-07-08,AMZN MKTP US*4K2LP,-58.99
2026-07-09,CITY WATER UTILITY,-64.10
2026-07-10,RENT PAYMENT - MAPLE APTS,-1450.00
```

## Next step: Phase 2

Build the self-correction loop: after the initial categorization pass, automatically
take every transaction marked `"confidence": "low"` and re-run it through a second,
stricter pass (more context, maybe asking the model to double-check against the full
list of categories and explain its reasoning) before showing final results. This is
the "loop engineering" piece — an agent checking its own work and revising, not just
running once.
