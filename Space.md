# Deploying to Spaces

- Space SDK: **Gradio**
- Python version: 3.10+

**Add `requirements.txt`** and these files:
- `app.py`
- `rag/` directory
- `prompts/system.txt`
- `data/processed/` with FAISS + parquet (upload small subset if repo size limits)

For large datasets, build the index locally and upload only the **processed** artifacts.
