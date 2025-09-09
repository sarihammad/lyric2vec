.PHONY: setup data prep train eval export faiss api dash up down logs test lint pipeline clean

# Setup environment
setup:
	pip install -r api/requirements.txt
	pip install -e ./core

# Data pipeline
data:
	python -m core.data.ingest --out data/raw

prep:
	python -m core.data.dataset --in data/raw --out data/processed

# Training
train:
	python -m core.train.ray_train --config core/train/config.yaml

eval:
	python -m core.train.eval --emb data/artifacts/embeddings.npz

# Indexing
export:
	python -m core.index.export_embeddings

faiss:
	python -m core.index.build_faiss --in data/artifacts/fused.npy --out data/artifacts/fused.index

# Services
api:
	uvicorn api.app.main:app --reload --port 8080

dash:
	streamlit run core/viz/dashboard.py

# Docker
up:
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f api

# Development
test:
	pytest -q

lint:
	ruff check .
	black --check .

# Full pipeline
pipeline: data prep train export faiss

# Cleanup
clean:
	rm -rf data/processed data/artifacts data/reports
