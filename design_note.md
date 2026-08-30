# Design Note: Lexical & Semantic Retrieval for News Recommendation

## 1. What was built and key design choices

I built a complete, reproducible pipeline for News Recommendation covering both the MIND and EB-NeRD datasets. The pipeline consists of the following components:
*   **Data Ingestion & Feature Store**: A robust data pipeline that downloads raw files, parses them into a unified Parquet schema, and performs temporal splits. A feature store was implemented for fast, leakage-safe lookups of article and user features.
*   **Lexical Candidate Generation**: BM25-based retrieval indexing article titles and abstracts.
*   **Semantic Candidate Generation**: Embedding-based retrieval using FAISS for Approximate Nearest Neighbor (ANN) search. For MIND, `sentence-transformers/all-MiniLM-L6-v2` embeddings were computed, while provided Word2Vec embeddings were used for EB-NeRD.
*   **Evaluation Harness**: An offline evaluation system computing AUC, MRR, nDCG@5, and nDCG@10, along with beyond-accuracy metrics (intra-list diversity, novelty, coverage). It includes bootstrap 95% confidence intervals and slicing capabilities.
*   **Codabench Submissions**: Highly optimized submission scripts generating prediction files (`prediction.txt` and `predictions.txt`) for the respective leaderboards.

**Key Design Choices:**
*   **Temporal Splitting & Dynamic Filtering**: To strictly enforce the behaviour-window boundary and prevent future-click leakage, click histories are dynamically filtered per-impression at query time via the feature store, rather than using a static pre-filtering approach.
*   **Vectorized Lexical Retrieval**: Chose the `bm25s` library over pure-Python implementations for BM25 to allow fast, batched scoring.
*   **Batch-Streaming for Submissions**: Used `pyarrow`'s batch-streaming API in the submission scripts to process millions of impressions without exceeding memory limits.

## 2. Alternatives considered and why you chose what you did

*   **Static Pre-filtering vs. Dynamic Query-time Filtering**: Initially, I attempted to filter user click histories using a static per-user global cutoff. This approach failed for the MIND train split, resulting in zero valid history rows. I transitioned to dynamic query-time filtering in `feature_store.py`, which correctly enforces the leakage boundary based on each impression's specific timestamp.
*   **BM25 Implementation (`rank_bm25` vs. `bm25s`)**: I originally implemented BM25 using the pure-Python `rank_bm25` library. However, this caused the evaluation script to effectively hang at MIND's scale (51k articles $\times$ thousands of queries). I switched to the vectorized `bm25s` library, which handles batched retrieval and resolved the performance bottleneck.
*   **Memory Management (Pandas vs. PyArrow)**: For the EB-NeRD submission script, loading all 13.5M test behavior rows into memory at once using Pandas caused local memory exhaustion and hanging. I opted for `pyarrow` batch-streaming to read and process the data in manageable chunks.
*   **Final Submission Scoring (BM25 vs. Embeddings)**: BM25 candidate scoring is computationally expensive for full-corpus ranking per query at the scale of 13.5M impressions. I chose embedding-based scoring (using FAISS) for the final Codabench submissions because it relies on highly optimized cosine similarity, scales much better, and generally showed stronger AUC/MRR performance in my offline evaluation.

## 3. Observations from experiments

*   **Lexical vs. Semantic Tradeoffs**: The winning method depends heavily on both the dataset and the number of retrieved candidates ($K$). 
*   **MIND Dataset**: Semantic retrieval (MiniLM embeddings) outperformed BM25 on AUC, MRR, nDCG, and Recall@K at $K \le 100$. However, BM25 catches up at $K=200$. This suggests that while semantic embeddings are better at ranking highly relevant items at the top, lexical retrieval might be more effective at finding niche matching candidates when retrieving a larger pool.
*   **EB-NeRD Dataset**: Interestingly, BM25 outperformed embeddings across all metrics on EB-NeRD. However, this is largely attributed to the *quality* of the embeddings rather than an inherent superiority of lexical retrieval: the EB-NeRD embeddings were based on Word2Vec (an older, weaker technique), whereas MIND utilized a modern transformer model (`all-MiniLM-L6-v2`).
*   **Ranking Signal Strength**: On the EB-NeRD demo dataset, the within-impression AUC for both methods sat close to 0.5 (with confidence intervals including 0.5) at the sampled size, indicating a very weak overall ranking signal in this specific data slice.

## 4. Where the pipeline breaks at 10x scale

*   **Lexical Candidate Scoring**: The current `score_candidates_batch` implementation in `bm25.py` performs a full-corpus ranking per query. While this is feasible for 11k-51k articles, scaling it 10x (e.g., for MIND-large's full impression volume) would be prohibitively slow. It would necessitate moving to an optimized inverted index with top-K pruning (such as the WAND algorithm) or offloading to a dedicated search engine like Elasticsearch.
*   **Cold-Start Evaluation Analysis**: The EB-NeRD demo bundle contains very few cold-start users (almost none under the <5 clicks threshold in my samples). A 10x scale-up to `ebnerd_small` or `ebnerd_large` would be required to perform statistically meaningful evaluations on cold-start users.
*   **MIND Timestamp Granularity**: MIND lacks exact per-click timestamps in the history column, so the pipeline uses the associated impression's timestamp. At a 10x scale, this coarse granularity might mask subtle chronological patterns, breaking advanced sequence-based recommendation models that require precise temporal ordering.
