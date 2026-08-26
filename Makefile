
.PHONY: data download parse split features index eval submit-mind submit-ebnerd clean help

DOWNLOAD_FLAGS ?=

help:
	@echo "Targets:"
	@echo "  make data          - full data pipeline: download + parse + split + feature store check"
	@echo "  make download      - download raw MIND + EB-NeRD data only; pass DOWNLOAD_FLAGS='...' for skips"
	@echo "                       example: make download DOWNLOAD_FLAGS='--include-ebnerd-small --include-embeddings'"
	@echo "  make parse         - parse raw data into unified schema (assumes already downloaded)"
	@echo "  make split         - temporal train/val/test split (assumes already parsed)"
	@echo "  make features      - feature store sanity check (assumes already split)"
	@echo "  make index         - build BM25 + embedding indexes, report Recall@K (Q2/Q3)"
	@echo "  make eval          - run full evaluation harness: AUC/MRR/nDCG + bootstrap + slicing (Q4)"
	@echo "  make submit-mind   - generate prediction.txt for MIND Codabench submission (Q5)"
	@echo "  make submit-ebnerd - generate predictions.txt for EB-NeRD Codabench submission (Q5)"
	@echo "  make clean         - remove all generated data (raw, processed, splits)"

data:
	python3 build_pipeline.py

download:
	python3 src/data/download.py $(DOWNLOAD_FLAGS)

parse:
	python3 build_pipeline.py --skip-download

split:
	python3 src/data/split.py

features:
	python3 src/data/feature_store.py

index:
	python3 src/retrieval/bm25.py --dataset both --sample-size 500
	python3 src/retrieval/ann_index.py --dataset both --sample-size 500

eval:
	python3 src/eval/run_eval.py --dataset both --sample-size 300

submit-mind:
	python3 src/submission/generate_predictions.py
	cd outputs/predictions && zip -j prediction.zip prediction.txt

submit-ebnerd:
	python3 src/submission/generate_predictions_ebnerd.py
	cd outputs/predictions && zip -j predictions.zip predictions.txt

clean:
	rm -rf data/raw data/processed data/splits