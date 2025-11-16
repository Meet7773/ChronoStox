import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import os

# ====================================================
# CONFIG
# ====================================================
NEWS_CSV = "news_dataset_clean.csv"
MASTER_SENTI = "sentiment_clean.parquet"
OUTPUT = "sentiment_clean.parquet"
MODEL_NAME = "ProsusAI/finbert"
BATCH = 128
DEBUG = True
# ====================================================


# ---------------- Load FinBERT ----------------------
def load_finbert():
    print("Loading FinBERT on CPU...")
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ---------------- Batch scoring ---------------------
def score_batch(texts, tokenizer, model, device):
    if not texts:
        return []

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    scores = probs[:, 0] - probs[:, 1]

    return scores.tolist()


# ====================================================
# MAIN PIPELINE
# ====================================================
def main():

    # 1) Load existing sentiment_clean
    if os.path.exists(MASTER_SENTI):
        print(f"Loading existing sentiment file: {MASTER_SENTI}")
        df_master = pd.read_parquet(MASTER_SENTI)
    else:
        df_master = pd.DataFrame(columns=["Date", "Ticker_YF", "sentiment_score"])

    # FIX MASTER FORMAT
    df_master["Date"] = pd.to_datetime(df_master["Date"], errors="coerce")
    df_master["Ticker_YF"] = df_master["Ticker_YF"].astype(str).str.upper()
    df_master = df_master.dropna(subset=["Date"])

    # Normalize master timezones
    df_master["Date"] = df_master["Date"].dt.tz_localize(None)

    # 2) Load news dataset
    df_news = pd.read_csv(NEWS_CSV)
    print(f"Loaded {len(df_news)} news rows")

    df_news["PublishedUTC"] = pd.to_datetime(df_news["PublishedUTC"], errors="coerce")
    df_news = df_news.dropna(subset=["PublishedUTC"])

    df_news["TickerNorm"] = df_news["TickerNorm"].astype(str).str.upper()

    # Always enforce .NS / .BO
    df_news["TickerNorm"] = df_news["TickerNorm"].apply(
        lambda t: t if t.endswith((".NS", ".BO")) else t + ".NS"
    )

    # Full text for FinBERT
    df_news["text"] = df_news["Title"].fillna("") + " " + df_news["Summary"].fillna("")

    # Remove rows already processed
    df_new = df_news.rename(columns={
        "PublishedUTC": "Date",
        "TickerNorm": "Ticker_YF"
    })[["Date", "Ticker_YF", "text"]]

    df_new = df_new[~df_new.set_index(["Date","Ticker_YF"]).index.isin(
        df_master.set_index(["Date","Ticker_YF"]).index
    )]

    print(f"New rows requiring sentiment scoring: {len(df_new)}")

    if len(df_new) == 0:
        print("Nothing new to score.")
        return

    # 3) Load FinBERT model
    tokenizer, model, device = load_finbert()

    # 4) Batch predict
    scores = []
    texts = df_new["text"].tolist()

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        batch_scores = score_batch(batch, tokenizer, model, device)
        scores.extend(batch_scores)

        if DEBUG:
            print(f"Scored {len(scores)}/{len(texts)}")

    df_new["sentiment_score"] = scores

    # Keep final ChronoStox format
    df_new = df_new[["Date", "Ticker_YF", "sentiment_score"]]

    # Normalize date
    df_new["Date"] = pd.to_datetime(df_new["Date"], errors="coerce")
    df_new["Date"] = df_new["Date"].dt.tz_localize(None)

    # 5) Merge with master
    df_final = pd.concat([df_master, df_new], ignore_index=True)

    df_final = df_final.drop_duplicates(subset=["Date", "Ticker_YF"], keep="last")
    df_final = df_final.sort_values(["Ticker_YF", "Date"]).reset_index(drop=True)

    # 6) Save
    df_final.to_parquet(OUTPUT, index=False)
    print(f"\n✅ Updated sentiment saved → {OUTPUT}")
    print(f"Total rows now: {len(df_final)}")

    print(df_final.tail(10))


if __name__ == "__main__":
    main()
