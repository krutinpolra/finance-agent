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
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Anthropic API key into .env
streamlit run app.py
```

## Deployment
Deployed automatically from the `main` branch via Streamlit Community Cloud.
Live demo: _(add link here once deployed)_
