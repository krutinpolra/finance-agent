"""
Personal Finance Agent — Phase 0 skeleton.

This file does NOT contain agent logic yet. Its only job right now is to prove
the full pipeline works: Streamlit renders -> reads your API key -> talks to
Claude -> shows a response. Once this works locally and deployed, Phase 1
replaces the button below with the real agent loop (CSV in, categorized
transactions out).
"""

import os
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # loads ANTHROPIC_API_KEY from .env when running locally

st.set_page_config(page_title="Personal Finance Agent", page_icon="💵")
st.title("Personal Finance Agent")
st.caption("Phase 0 — connectivity check. Agent logic lands in Phase 1.")


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
st.subheader("Step 2: coming in Phase 1")
st.file_uploader(
    "Upload a transactions CSV (not wired up yet — placeholder for next phase)",
    type="csv",
    disabled=True
)
