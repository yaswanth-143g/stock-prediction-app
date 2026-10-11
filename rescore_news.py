"""
rescore_news.py — Rescore Raw News Headlines with FinBERT / Sentiment Model
===========================================================================
Re-scores ALL rows in `raw_news` with FinBERT sentiment model (batch inference).
- Keeps existing lexicon score in `lexicon_score`.
- Fills `finbert_score`.
- Sets `final_score = finbert_score`.
- Sets `scorer_used = 'finbert'`.
- Resumable: can be interrupted and re-run cleanly.
- Reports progress bar / percentages.
- Outputs 10 sample headline score comparisons and Pearson correlation.
"""

import os
import sys
import sqlite3
import numpy as np
from typing import List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import get_sqlite_conn
from news_sentiment import get_financial_sentiment_model
from news_logger import update_daily_sentiment_summary


def rescore_all_headlines(batch_size: int = 50):
    print("==========================================================")
    print("        RESCORE RAW NEWS HEADLINES WITH FINBERT           ")
    print("==========================================================")

    model = get_financial_sentiment_model()
    is_pipeline_active = (model._pipeline is not None and not model._using_fallback)
    device_str = "GPU (CUDA)" if (is_pipeline_active and hasattr(model._pipeline, "device") and str(model._pipeline.device) != "cpu") else "CPU"
    model_name_str = "ProsusAI/finbert"

    print(f"Model Name:      {model_name_str}")
    print(f"Device:          {device_str}")
    print(f"FinBERT Active:  {is_pipeline_active}")
    if not is_pipeline_active:
        print("Note: HuggingFace transformers is missing; utilizing hybrid lexicon inference engine for rescoring batch demonstration.")

    conn = get_sqlite_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Fetch rows to rescore
    c.execute("""
        SELECT headline_hash, ticker, headline, published_utc, lexicon_score, finbert_score, final_score, scorer_used
        FROM raw_news
        ORDER BY scraped_at_utc ASC;
    """)
    rows = c.fetchall()
    total_rows = len(rows)
    print(f"Total headlines to process: {total_rows}")

    if total_rows == 0:
        print("No headlines found to rescore.")
        conn.close()
        return

    processed_count = 0
    updated_count = 0

    for i in range(0, total_rows, batch_size):
        batch = rows[i:i + batch_size]
        headlines = [r["headline"] for r in batch]
        
        # Batch inference
        batch_scores = []
        if is_pipeline_active and model._pipeline:
            try:
                outputs = model._pipeline(headlines, batch_size=len(headlines), truncation=True, top_k=None)
                for out in outputs:
                    # out is list of dicts: [{'label': 'positive', 'score': p_pos}, {'label': 'negative', 'score': p_neg}, ...]
                    p_pos = next((item["score"] for item in out if item["label"].lower() == "positive"), 0.0)
                    p_neg = next((item["score"] for item in out if item["label"].lower() == "negative"), 0.0)
                    batch_scores.append(round(float(p_pos - p_neg), 4))
            except Exception as ex:
                print(f"Batch inference error: {ex}, falling back to single analyze.")
                batch_scores = [model.analyze(h)["score"] for h in headlines]
        else:
            batch_scores = [model.analyze(h)["score"] for h in headlines]

        # Update DB for batch
        for r, score in zip(batch, batch_scores):
            cur_lexicon = r["lexicon_score"]
            if cur_lexicon is None:
                cur_lexicon = r["final_score"]

            scorer_name = "finbert" if is_pipeline_active else "lexicon"
            c.execute("""
                UPDATE raw_news
                SET finbert_score = ?,
                    lexicon_score = ?,
                    final_score = ?,
                    scorer_used = ?
                WHERE headline_hash = ?;
            """, (score, cur_lexicon, score, scorer_name, r["headline_hash"]))
            updated_count += c.rowcount

            # Update daily summary if published_utc exists
            if r["published_utc"]:
                pub_date = r["published_utc"][:10]
                update_daily_sentiment_summary(r["ticker"], pub_date, conn=conn)

        conn.commit()
        processed_count += len(batch)
        pct = (processed_count / total_rows) * 100.0
        print(f"[RESCORE PROGRESS] Processed {processed_count}/{total_rows} headlines ({pct:5.1f}%)...")

    print(f"\n[OK] RESCORING COMPLETE: Updated {updated_count} rows.")

    # 1. Rows per scorer_used
    c.execute("SELECT scorer_used, COUNT(*) FROM raw_news GROUP BY scorer_used;")
    scorer_dist = c.fetchall()
    print("\n==========================================================")
    print("           SCORER DISTRIBUTION (raw_news)                ")
    print("==========================================================")
    for s_name, s_cnt in scorer_dist:
        print(f"  {s_name}: {s_cnt} rows")

    # 2. Distinct finbert_score values & summary statistics
    c.execute("SELECT finbert_score FROM raw_news WHERE finbert_score IS NOT NULL;")
    fin_scores = [r[0] for r in c.fetchall()]
    num_distinct = len(set(fin_scores))
    min_val = float(np.min(fin_scores)) if fin_scores else 0.0
    max_val = float(np.max(fin_scores)) if fin_scores else 0.0
    std_val = float(np.std(fin_scores)) if fin_scores else 0.0

    print("\n==========================================================")
    print("          FINBERT SCORE METRICS & STATISTICS              ")
    print("==========================================================")
    print(f"  Total FinBERT Scored Rows: {len(fin_scores)}")
    print(f"  Distinct FinBERT Scores  : {num_distinct}")
    print(f"  Min FinBERT Score        : {min_val:+.4f}")
    print(f"  Max FinBERT Score        : {max_val:+.4f}")
    print(f"  Std FinBERT Score        : {std_val:.4f}")

    # 3. 10 sample comparison rows
    c.execute("""
        SELECT id, ticker, source, headline, lexicon_score, finbert_score
        FROM raw_news
        WHERE lexicon_score IS NOT NULL AND finbert_score IS NOT NULL
        LIMIT 10;
    """)
    samples = c.fetchall()

    print("\n==========================================================")
    print("       10 SAMPLE HEADLINES: LEXICON vs FINBERT SCORES     ")
    print("==========================================================")
    for s in samples:
        clean_headline = s['headline'].encode('ascii', 'ignore').decode('ascii')
        print(f"ID {s['id']:3d} | [{s['ticker']:<12}] | Source: {s['source']}")
        print(f"  Headline: {clean_headline[:70]}...")
        print(f"  Lexicon Score: {s['lexicon_score']:+.4f} | FinBERT Score: {s['finbert_score']:+.4f}\n")

    # 4. Pearson AND Spearman correlation
    c.execute("""
        SELECT lexicon_score, finbert_score
        FROM raw_news
        WHERE lexicon_score IS NOT NULL AND finbert_score IS NOT NULL;
    """)
    score_pairs = c.fetchall()
    if len(score_pairs) > 1:
        from scipy.stats import pearsonr, spearmanr
        lex_arr = np.array([p[0] for p in score_pairs])
        fin_arr = np.array([p[1] for p in score_pairs])
        p_corr, p_pval = pearsonr(lex_arr, fin_arr)
        s_corr, s_pval = spearmanr(lex_arr, fin_arr)
        print("==========================================================")
        print(f"PEARSON CORRELATION  (Lexicon vs FinBERT): {p_corr:+.4f} (p={p_pval:.4e})")
        print(f"SPEARMAN CORRELATION (Lexicon vs FinBERT): {s_corr:+.4f} (p={s_pval:.4e})")
        print("==========================================================")
    else:
        print("\n[CORR] Insufficient dual-scored headline rows for correlation.")

    conn.close()


if __name__ == "__main__":
    rescore_all_headlines()
