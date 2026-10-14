Micro-Enterprise Financial & Operational Advisory (CP5403 Assessment 2)
RAG + agent workflow (+ optional LoRA fine-tune) for micro-enterprise questions.
Setup
pip install -r requirements.txt
python scripts/ingest.py     # downloads sources in data/sources.yaml, builds data/processed/chunks.jsonl
Layout
data/sources.yaml   corpus manifest (provenance + licences)
scripts/ingest.py   download, extract, chunk
src/                pipeline code (next stage)
eval/               test set + evaluation code
