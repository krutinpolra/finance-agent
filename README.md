# Personal Finance Agent

An AI agent that reads bank/credit-card statements — in whatever format they show up in —
categorizes every transaction, catches and rechecks its own low-confidence guesses, and
hands the result to a second agent that reviews the numbers and gives budget advice.

Built from scratch (no LangChain/CrewAI/agent framework) specifically to learn how agent
loops and multi-agent systems actually work under the hood, not just how to call one.

## The problem

Categorizing spending from a real statement is tedious and the input is never clean: one
person's statement is a CSV export, another's is a PDF from their bank, someone else has a
photo of a receipt or a Word doc. Rule-based categorizers break the moment a merchant name
doesn't match a known pattern, and naive "ask an LLM once" approaches don't have any
mechanism to notice when they're guessing. This project explores a more deliberate design:
a cheap deterministic lookup for the easy 80%, model judgment for the rest, a second pass
that specifically targets the cases the model itself flagged as uncertain, and a downstream
agent that turns the result into something a person can actually act on.

## What it does

- Accepts one or more statements per upload — **CSV, PDF, PNG/JPG, or DOCX** — and merges
  them into a single transaction table.
- Categorizes every transaction with a tool-calling agent: it checks a local
  known-merchant lookup first (instant, free, deterministic) and only falls back to model
  judgment when the lookup doesn't have an answer.
- **Self-corrects**: anything the agent itself marks low confidence gets a second, stricter
  pass — with category definitions and a requirement to reason before answering — instead
  of being shown to the user as-is.
- Hands the categorized summary to a second agent, a **Budget Advisor**, which checks each
  category's spend against typical budgeting benchmarks (via its own tool) and writes a
  plain-English recommendation.
- Includes a hand-labeled evaluation set (`eval_data/`) and a script (`eval.py`) that
  measures categorization accuracy against it end-to-end — run `python eval.py` to
  reproduce.

## How it works

```
statement(s) (csv/pdf/image/docx)
        │
        ▼
  Extraction agent  ──  reads the file directly (Claude's native PDF/image
        │                input, or extracted text for docx) and returns
        │                structured {date, description, amount} rows
        ▼
  Categorizer agent  ──  tool-calling loop: known-merchant lookup first,
        │                model judgment as fallback, flags its own
        │                low-confidence guesses
        ▼
  Self-correction pass  ──  re-runs only the low-confidence rows through a
        │                    stricter prompt before finalizing
        ▼
  Budget Advisor agent  ──  reviews the categorized totals against spending
                             benchmarks (its own tool) and writes advice
```

Every agent here is the same underlying mechanism — call the model, check if it asked to
use a tool, run the tool, feed the result back, repeat until it gives a final answer —
specialized by system prompt and tool access. Writing that loop by hand once, then reusing
the pattern across four different agents, was the actual point of the project.

## Tech stack

- **Python + Streamlit** — UI and backend in one file, so the focus stays on agent logic
- **Anthropic Claude API** (`claude-haiku-4-5`) — categorization, self-correction, document/
  image extraction, and budget advice
- **pandas** for tabular handling, **python-docx** for Word doc text extraction
- **GitHub + Streamlit Community Cloud** for free hosting

## Status

- [x] Hand-rolled agent loop with real tool-calling (categorization)
- [x] Self-correction pass on low-confidence results
- [x] Second agent (Budget Advisor) — first multi-agent handoff
- [x] Multi-format, multi-file ingestion (CSV / PDF / image / DOCX via native
      vision/document input)
- [x] Hand-labeled eval set + accuracy script
- [ ] Cost/latency logging per run
- [ ] Deployed live demo
- [ ] Write-up on design decisions

## Running locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Anthropic API key into .env
streamlit run app.py
```

No file handy? Use the sample at `sample_data/transactions.csv`.

## Evaluating accuracy

```bash
python eval.py
```

Runs the full categorize → self-correct pipeline against `eval_data/labeled_transactions.csv`
(24 hand-labeled transactions spanning all 12 categories, including a couple of
deliberately ambiguous ones) and reports accuracy plus every miss.

## Known limitations

- PDF/image extraction is best-effort — quality depends on how clean the source scan or
  photo is; it isn't OCR-perfect.
- No persistence yet — each upload is processed in-session, nothing is stored between visits.

## Deployment

Live demo: _(add link here once deployed)_
