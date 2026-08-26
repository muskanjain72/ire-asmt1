# ire-asmt1

prediction.zip (MIND) and predictions.zip (EB-NeRD)

# CS4.406 Assignment 1: Lexical & Semantic Retrieval on EB-NeRD and MIND

News recommendation pipeline covering lexical (BM25) and semantic (embedding-based) candidate
retrieval, an offline evaluation harness, and Codabench leaderboard submissions for both the
MIND and EB-NeRD (RecSys 2024) challenges.

## Results Summary

### Leaderboard Submissions

| Competition | Rank | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MIND (Codabench 13967) | 31 / 37 | 0.6501 | 0.3183 | 0.3434 | 0.3993 |
| EB-NeRD / RecSys 2024 (Codabench 2469) | 5 | 0.5139 | 0.3241 | 0.3577 | 0.4425 |

### Recall@K — Candidate Generation (Q2/Q3)

| Dataset | Method | R@50 | R@100 | R@200 |
|---|---|---|---|---|
| MIND | BM25 | 0.0143 | 0.0179 | 0.0502 |
| MIND | Embeddings (MiniLM) | 0.0143 | 0.0251 | 0.0358 |
| EB-NeRD | BM25 | 0.0200 | 0.0300 | 0.0460 |
| EB-NeRD | Embeddings (Word2Vec) | 0.0080 | 0.0100 | 0.0220 |

### Real Evaluation Harness — AUC/MRR/nDCG (Q4, sample_size=150 impressions)

| Dataset | Method | AUC (95% CI) | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MIND | BM25 | 0.568 [0.500, 0.637] | 0.299 | 0.264 | 0.329 |
| MIND | Embeddings | 0.677 [0.617, 0.748] | 0.429 | 0.401 | 0.428 |
| EB-NeRD | BM25 | 0.469 [0.424, 0.517] | 0.263 | 0.285 | 0.375 |
| EB-NeRD | Embeddings | 0.473 [0.423, 0.525] | 0.280 | 0.305 | 0.396 |

### Beyond-Accuracy (top-20 retrieval)

| Dataset | Method | Diversity | Novelty | Coverage |
|---|---|---|---|---|
| MIND | BM25 | 0.768 | 14.71 | 0.024 (1218/51282) |
| MIND | Embeddings | 0.611 | 14.48 | 0.021 (1099/51282) |
| EB-NeRD | BM25 | 0.167 | 9.79 | 0.087 (1023/11777) |
| EB-NeRD | Embeddings | 0.049 | 11.84 | 0.019 (227/11777) |

**Key finding**: which method wins depends on both dataset and K. On MIND, embeddings
(sentence-transformers `all-MiniLM-L6-v2`) beat BM25 on AUC/MRR/nDCG and at K≤100 recall, but
BM25 catches up at K=200. On EB-NeRD, BM25 outperforms embeddings across the board — but the
EB-NeRD embeddings used are the dataset's provided **Word2Vec** vectors (an older, weaker
technique than MiniLM), so this is more a finding about embedding *quality* than semantic vs.
lexical retrieval in general. EB-NeRD's within-impression AUC for both methods sits close to
0.5 with CIs that include 0.5 — the ranking signal is weak on this dataset at this sample size.

## Project Structure

```
.
├── Makefile                    # one-command entrypoints for every stage
├── build_pipeline.py           # Q1 orchestrator: download -> parse -> split -> feature store
├── requirements.txt
├── data/
│   ├── raw/                    # downloaded MIND + EB-NeRD files (gitignored)
│   ├── processed/              # unified-schema parquet per dataset (gitignored)
│   └── splits/                 # temporal train/val/test + feature store files (gitignored)
├── src/
│   ├── data/
│   │   ├── download.py         # automated dataset download (Kaggle for MIND, S3 for EB-NeRD)
│   │   ├── parse_mind.py       # MIND raw -> unified schema
│   │   ├── parse_ebnerd.py     # EB-NeRD raw -> unified schema
│   │   ├── split.py            # temporal (never random) train/val/test split
│   │   └── feature_store.py    # leakage-safe article + user feature lookups
│   ├── retrieval/
│   │   ├── bm25.py             # Q2: BM25 inverted index + Recall@K eval
│   │   ├── embeddings.py       # Q3: load/compute article embeddings
│   │   └── ann_index.py        # Q3: FAISS ANN index + Recall@K eval
│   ├── eval/
│   │   ├── metrics.py          # Q4: AUC, MRR, nDCG@5/10
│   │   ├── beyond_accuracy.py  # Q4: diversity, novelty, coverage
│   │   ├── bootstrap.py        # Q4: bootstrap 95% CIs
│   │   ├── slicing.py          # Q4: cold-start/warm, head/tail slicing
│   │   └── run_eval.py         # Q4: wires real BM25/embedding scores into the above
│   └── submission/
│       ├── generate_predictions.py         # Q5: MIND -> prediction.txt
│       └── generate_predictions_ebnerd.py  # Q5: EB-NeRD -> predictions.txt
├── tests/
│   └── test_no_leakage.py      # Q9: asserts no future-click leakage
└── outputs/predictions/        # final submission files (see below)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two external credentials are needed before downloading data:
- **Kaggle API token** (for MIND): create at https://www.kaggle.com/settings → API → "Create New
  Token", save as `~/.kaggle/kaggle.json`, `chmod 600` it.
- No credentials needed for EB-NeRD (public S3 bucket).

## Reproduce Everything (One Command)

```bash
make data     # Q1: download + parse + split + feature store — full pipeline from raw files
make index    # Q2/Q3: build BM25 + embedding indexes, report Recall@K
make eval     # Q4: full evaluation harness — AUC/MRR/nDCG, bootstrap CIs, slicing
```

Run `make help` to see every available target.

## Reproduce Individual Stages

```bash
# Q1 — data pipeline
python3 src/data/download.py                  # download raw files
python3 build_pipeline.py --skip-download      # parse + split + feature store (reuses local data/raw/)

# Q2 — BM25
python3 src/retrieval/bm25.py --dataset both --sample-size 500

# Q3 — Embeddings
python3 src/retrieval/embeddings.py            # computes/caches article embeddings
python3 src/retrieval/ann_index.py --dataset both --sample-size 500

# Q4 — Evaluation harness
python3 src/eval/run_eval.py --dataset both --sample-size 150

# Q9 — Leakage test
pytest tests/test_no_leakage.py -v
```

## Generating Submission Files (Q5)

```bash
# MIND — writes outputs/predictions/prediction.txt (singular filename, per MIND's spec)
python3 src/submission/generate_predictions.py --test-limit 1000   # smoke test first
python3 src/submission/generate_predictions.py                     # full run
cd outputs/predictions && zip -j prediction.zip prediction.txt

# EB-NeRD — writes outputs/predictions/predictions.txt (plural filename, per EB-NeRD's spec)
python3 src/submission/generate_predictions_ebnerd.py --test-limit 1000
python3 src/submission/generate_predictions_ebnerd.py
cd outputs/predictions && zip -j predictions.zip predictions.txt
```

**Note the filename difference is deliberate and required by each competition's own spec:**
- `prediction.txt` (singular) → MIND, uploaded as `prediction.zip`
- `predictions.txt` (plural) → EB-NeRD, uploaded as `predictions.zip`

Both scripts use embedding-based scoring (not BM25) for the final submission, since Q4's
real evaluation showed embeddings outperforming BM25 on the within-impression ranking task
that these leaderboards actually score (see AUC/MRR/nDCG table above).

## Datasets

| | MIND | EB-NeRD |
|---|---|---|
| Train/dev used | MINDsmall (via Kaggle mirror) | ebnerd_demo |
| Test set used | MINDlarge_test (MINDsmall has no official test split) | ebnerd_testset |
| Articles | 51,282 (train) / 120,961 (test) | 11,777 (train) / 125,541 (test) |
| Embeddings | Computed: `sentence-transformers/all-MiniLM-L6-v2` | Provided: Ekstra Bladet Word2Vec |

**Known limitation**: MIND's official test set only exists at the "large" scale — there is no
MINDsmall test split — so submission-time scoring uses a freshly-built index over
MINDlarge_test's own articles rather than the MINDsmall index used for local dev/eval.

## Known Limitations / Where This Breaks at 10x

- **BM25 candidate scoring at test-submission scale**: `bm25.py`'s `score_candidates_batch`
  does a full-corpus ranking per query, which is fine at 11k-51k articles but would not scale
  to MIND-large's full impression volume — this is why the final Codabench submissions use
  embedding-based scoring (cheap cosine similarity) rather than BM25.
- **EB-NeRD demo bundle has limited cold-start users**: only ~1,935 of its users have any click
  history at all, and none fall under the cold-start threshold (<5 clicks) in our sampled
  evaluations — meaningful cold-start analysis would require `ebnerd_small` or `ebnerd_large`.
- **MIND's click-history timestamps are upper-bounded, not exact**: MIND's `history` column has
  no per-click timestamp, so `parse_mind.py` stamps each history entry with its associated
  impression's own timestamp — slightly coarser than EB-NeRD's true per-click timestamps.

## AI Usage

See `ai_usage_log.md` for the full prompt log and AI-generated vs. human-written/verified
code breakdown.

## report link

https://docs.google.com/document/d/1_UMaYD-9Sz3AhBJuSuny-RJFtFouMLOS9ph5M9nDFKk/edit?usp=sharing
