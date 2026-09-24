#!/usr/bin/env python3
"""
Download and preprocess a public spam dataset into train/eval CSVs.

Uses the 'sms_spam' dataset from HuggingFace (SMS Spam Collection) as a
readily available, permissively licensed baseline. For a larger email-domain
dataset, swap the loader for Enron-Spam (see commented section below).

Output:
    data/train.csv   columns: text, label  (0=ham, 1=spam)
    data/eval.csv    columns: text, label

Usage:
    python prepare_data.py
    python prepare_data.py --dataset enron   # use Enron-Spam instead
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str, max_chars: int = 512) -> str:
    """Strip HTML, collapse whitespace, truncate to max_chars."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        text = ""
    text = _HTML_TAG.sub(" ", str(text))
    text = _WHITESPACE.sub(" ", text).strip()
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_sms_spam() -> pd.DataFrame:
    """
    HuggingFace 'sms_spam' dataset (~5,500 SMS messages).
    Label mapping: 'ham' -> 0, 'spam' -> 1.
    Text field: 'sms' (no subject — used as-is).
    """
    ds = load_dataset("sms_spam", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    df = df.rename(columns={"sms": "text"})
    # label column is already 0/1 in this dataset
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]]


def load_enron() -> pd.DataFrame:
    """
    HuggingFace 'SetFit/enron_spam' dataset (~33,000 emails).
    Uses subject + body concatenation to match training input format.
    """
    ds = load_dataset("SetFit/enron_spam", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    subject = df.get("subject", pd.Series([""] * len(df))).fillna("")
    body = df.get("body", pd.Series([""] * len(df))).fillna("")
    df["text"] = (subject + " " + body).apply(clean_text)
    # label_num: 0=ham, 1=spam
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


LOADERS = {
    "sms": load_sms_spam,
    "enron": load_enron,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(LOADERS.keys()),
        default="enron",
        help="Which public dataset to use (default: enron)",
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for eval (default: 0.2)",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Directory to write train.csv and eval.csv (default: data/)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{args.dataset}' dataset...")
    df = LOADERS[args.dataset]()

    # Drop rows with empty/whitespace-only text (they fall back to NaN on CSV round-trip)
    before = len(df)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with empty text")

    print(f"Total examples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().rename({0: 'ham', 1: 'spam'})}\n")

    spam_frac = df["label"].mean()
    if spam_frac < 0.3 or spam_frac > 0.7:
        print(
            f"⚠️  Imbalanced dataset ({spam_frac:.1%} spam). "
            "Consider passing class_weight to the Trainer."
        )

    train_df, eval_df = train_test_split(
        df,
        test_size=args.eval_size,
        stratify=df["label"],
        random_state=42,
    )

    train_path = out / "train.csv"
    eval_path = out / "eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"Train: {len(train_df)} rows  →  {train_path}")
    print(f"Eval:  {len(eval_df)} rows  →  {eval_path}")
    print(
        f"Train label split: "
        f"{(train_df['label'] == 0).sum()} ham / {(train_df['label'] == 1).sum()} spam"
    )


if __name__ == "__main__":
    main()
