import os
import zipfile
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

from feature_store import FeatureStore
from bm25 import BM25Index
from ann_index import ANNIndex, user_vector, score_candidates_batch
from embeddings import get_or_compute_embeddings

def generate_predictions(test_impressions_path: str, dataset_name: str, output_zip: str):
    """
    Generates a zip file containing predictions.txt for the Codabench submission.
    """
    if not os.path.exists(test_impressions_path):
        print(f"Error: Could not find test impressions at {test_impressions_path}")
        print("Please make sure you have downloaded the ebnerd_testset and extracted it.")
        return

    print(f"Loading test impressions from {test_impressions_path}...")
    # The test set might use "article_ids_inview" or "article_id" depending on how it's formatted.
    # We will try to handle both.
    df = pd.read_parquet(test_impressions_path)
    
    # Check if the dataframe is exploded or grouped
    if "article_id" in df.columns and "impression_id" in df.columns:
        print("Grouping exploded impressions...")
        # If it's exploded like the training set
        grouped = df.groupby("impression_id")
        impressions = []
        for imp_id, group in tqdm(grouped, desc="Grouping"):
            impressions.append({
                "impression_id": imp_id,
                "user_id": group["user_id"].iloc[0],
                "timestamp": group["timestamp"].iloc[0],
                "article_ids_inview": group["article_id"].tolist()
            })
    else:
        # If it's already grouped (as described in the Codabench instructions: "article_ids_inview")
        impressions = df.to_dict("records")
        if "article_ids_inview" not in df.columns and "article_ids" in df.columns:
            for imp in impressions:
                imp["article_ids_inview"] = imp["article_ids"]

    print(f"Total impressions: {len(impressions)}")
    
    print("Loading feature store and embeddings...")
    # Initialize the FeatureStore using the training/val data splits to access user history
    store = FeatureStore("data/splits", dataset_name)
    embeddings = get_or_compute_embeddings(dataset_name, store)

    print("Generating predictions...")
    
    # We will score candidates using the embeddings approach (which is usually the primary model)
    # Alternatively, you can use BM25 if you prefer by instantiating BM25Index
    
    predictions_text = []
    
    # Process in batches for efficiency
    batch_size = 1000
    for i in tqdm(range(0, len(impressions), batch_size), desc="Scoring"):
        batch = impressions[i:i+batch_size]
        
        query_vectors = []
        candidate_lists = []
        for imp in batch:
            hist_ids = store.get_user_history(imp["user_id"], imp["timestamp"])
            q_vec = user_vector(hist_ids, embeddings)
            query_vectors.append(q_vec)
            candidate_lists.append(imp["article_ids_inview"])
            
        # Score candidates
        score_dicts = score_candidates_batch(query_vectors, candidate_lists, embeddings)
        
        for imp, score_dict in zip(batch, score_dicts):
            imp_id = imp["impression_id"]
            candidates = imp["article_ids_inview"]
            
            # If we couldn't score (e.g. cold start user with no history), we provide a random/default ranking
            if not score_dict or all(v == 0.0 for v in score_dict.values()):
                scores = [0.0] * len(candidates)
            else:
                scores = [score_dict.get(aid, 0.0) for aid in candidates]
                
            # We want to rank them where the highest score gets rank 1, second gets 2, etc.
            # np.argsort sorts ascending. We do negative to sort descending.
            # To get the rank of each item in the original list:
            
            # Example: scores = [0.9, 0.1, 0.8, 0.95]
            # argsort(-scores) -> [3, 0, 2, 1]  (indices of items from highest to lowest score)
            # We need to output the rank (1-indexed) for each original position.
            # rank_array[3] = 1, rank_array[0] = 2, rank_array[2] = 3, rank_array[1] = 4
            
            neg_scores = -np.array(scores)
            sorted_indices = np.argsort(neg_scores)
            
            ranks = np.zeros(len(scores), dtype=int)
            for rank_idx, original_idx in enumerate(sorted_indices):
                ranks[original_idx] = rank_idx + 1  # 1-indexed rank
                
            rank_list_str = "[" + ",".join(map(str, ranks)) + "]"
            predictions_text.append(f"{imp_id} {rank_list_str}")

    print("Writing predictions.txt...")
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    
    txt_path = os.path.join(os.path.dirname(output_zip), "predictions.txt")
    with open(txt_path, "w") as f:
        for line in predictions_text:
            f.write(line + "\n")
            
    print(f"Creating zip file: {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Codabench requires the file to be exactly 'predictions.txt' inside the zip, without any directories
        zipf.write(txt_path, arcname="predictions.txt")
        
    print("Done! You can now submit the zip file to Codabench.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # You will need to change this path to wherever you extract the test set
    parser.add_argument("--test-impressions", type=str, default="data/raw/ebnerd_testset/test/impressions.parquet", help="Path to the testset impressions parquet file")
    parser.add_argument("--dataset", type=str, default="ebnerd", help="Dataset name to load embeddings (ebnerd/mind)")
    parser.add_argument("--output", type=str, default="outputs/predictions/predictions_ebnerd_test.zip", help="Output zip file path")
    args = parser.parse_args()
    
    generate_predictions(args.test_impressions, args.dataset, args.output)
