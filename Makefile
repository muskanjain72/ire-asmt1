.PHONY: data download parse split features clean help

help:
	@echo "Targets:"
	@echo "  make data      - full pipeline: download + parse + split + feature store check"
	@echo "  make download  - download raw MIND + EB-NeRD data only"
	@echo "  make parse     - parse raw data into unified schema (assumes already downloaded)"
	@echo "  make split     - temporal train/val/test split (assumes already parsed)"
	@echo "  make features  - feature store sanity check (assumes already split)"
	@echo "  make clean     - remove all generated data (raw, processed, splits)"

data:
	python3 build_pipeline.py

download:
	python3 src/data/download.py

parse:
	python3 build_pipeline.py --skip-download

split:
	python3 src/data/split.py

features:
	python3 src/data/feature_store.py

clean:
	rm -rf data/raw data/processed data/splits