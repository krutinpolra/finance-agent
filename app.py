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
