"""
Phase 4 — accuracy eval.

Runs the full pipeline (categorize, then a stricter recheck on anything low
confidence - same rule the Streamlit app follows) against a hand-labeled set
of transactions in eval_data/labeled_transactions.csv, and reports what
percentage matched the expected category.

This is what turns "I built an agent" into "I measured whether it works" -
a labeled set is a human-decided answer key, separate from the agent's own
judgment, that the agent's output gets checked against.

Run directly: `python eval.py`
"""

import os
import time

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from agent import categorize_transaction

load_dotenv()


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY in your .env file first.")
    client = Anthropic(api_key=api_key)

    df = pd.read_csv("eval_data/labeled_transactions.csv")

    predicted, confidences = [], []
    start = time.perf_counter()

    for n, row in df.iterrows():
        result = categorize_transaction(client, row["description"], row["amount"])
        if result["confidence"] == "low":
            result = categorize_transaction(
                client, row["description"], row["amount"], strict=True
            )
        predicted.append(result["category"])
        confidences.append(result["confidence"])
        print(f"[{n + 1}/{len(df)}] {row['description'][:35]:<35} -> {result['category']}")

    elapsed = time.perf_counter() - start

    df["predicted_category"] = predicted
    df["confidence"] = confidences
    df["correct"] = df["predicted_category"] == df["expected_category"]

    accuracy = df["correct"].mean() * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({df['correct'].sum()}/{len(df)}) in {elapsed:.1f}s\n")

    misses = df[~df["correct"]]
    if len(misses) > 0:
        print("Misses:")
        for _, row in misses.iterrows():
            print(
                f"  '{row['description']}' -> got '{row['predicted_category']}', "
                f"expected '{row['expected_category']}'"
            )


if __name__ == "__main__":
    main()
