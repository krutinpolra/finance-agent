"""
Personal Finance Agent — Phase 3.

Step 1 is the same connectivity check from Phase 0. Step 2 runs the agent loop
(see agent.py) over every row of an uploaded CSV, then a self-correction pass
re-checks anything flagged low confidence. Step 3 hands the categorized
summary to a second agent (see budget_advisor.py) that reviews it and gives
advice — one agent's output becomes another agent's input.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from agent import categorize_transaction
from budget_advisor import get_budget_advice
from extract import extract_transactions

load_dotenv()  # loads ANTHROPIC_API_KEY from .env when running locally

st.set_page_config(page_title="Personal Finance Agent", page_icon="💵")
st.title("Personal Finance Agent")
st.caption("Phase 3 — categorization + self-correction, reviewed by a second Budget Advisor agent.")


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

uploaded_files = st.file_uploader(
    "Upload one or more bank/credit-card statements — CSV, PDF, image, or Word doc",
    type=["csv", "pdf", "png", "jpg", "jpeg", "docx"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Categorize transactions"):
    client = get_client()

    extracted = []
    extract_progress = st.progress(0, text="Starting...")
    for n, file in enumerate(uploaded_files):
        file_df = extract_transactions(client, file)
        if not file_df.empty:
            file_df["source_file"] = file.name
            extracted.append(file_df)
        extract_progress.progress(
            (n + 1) / len(uploaded_files),
            text=f"Read {n + 1}/{len(uploaded_files)}: {file.name}"
        )
    extract_progress.empty()

    df = pd.concat(extracted, ignore_index=True) if extracted else pd.DataFrame(columns=["date", "description", "amount"])

    if df.empty:
        st.warning("Couldn't find any transactions in those files. Try clearer scans/photos, or a CSV.")
    else:
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
        df["reasoning"] = ""
        df["rechecked"] = False
        progress.empty()

        # Self-correction pass: anything the agent itself was unsure about gets a
        # second, closer look before we show final results.
        low_confidence = df.index[df["confidence"] == "low"]
        if len(low_confidence) > 0:
            recheck_progress = st.progress(0, text="Double-checking low-confidence transactions...")
            for n, i in enumerate(low_confidence):
                row = df.loc[i]
                result = categorize_transaction(
                    client, row["description"], row["amount"], strict=True
                )
                df.loc[i, "category"] = result["category"]
                df.loc[i, "confidence"] = result["confidence"]
                df.loc[i, "reasoning"] = result.get("reasoning", "")
                df.loc[i, "rechecked"] = True
                recheck_progress.progress(
                    (n + 1) / len(low_confidence),
                    text=f"Double-checked {n + 1}/{len(low_confidence)}: {row['description'][:30]}"
                )
            recheck_progress.empty()
            st.info(f"{len(low_confidence)} transaction(s) needed a second look.")

        # Stash in session_state: Streamlit reruns this whole script on every
        # button click, and the "Categorize transactions" button is only True on
        # the run where it was actually clicked. Without saving df here, clicking
        # the Step 3 button below would rerun the script, find that button False
        # again, and the categorized data would disappear before Step 3 could use it.
        st.session_state["categorized_df"] = df

if "categorized_df" in st.session_state:
    df = st.session_state["categorized_df"]
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.groupby("category")["amount"].sum().abs())

    st.divider()
    st.subheader("Step 3: get budget advice")

    spend_by_category = df[df["category"] != "Income"].groupby("category")["amount"].sum().abs()
    income_total = df.loc[df["category"] == "Income", "amount"].sum()

    if income_total <= 0:
        st.caption("Add an 'Income' transaction to your CSV to unlock budget advice.")
    elif st.button("Get budget advice"):
        client = get_client()
        category_percentages = (spend_by_category / income_total * 100).to_dict()
        with st.spinner("Reviewing your spending..."):
            advice = get_budget_advice(client, category_percentages)
        st.markdown(advice)
