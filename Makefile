# -----------------------------
# BakeChat RAG — Makefile
# -----------------------------
# Targets you'll actually use:
#   make install         Create venv + install requirements
#   make build-index     Build FAISS index from data/raw -> data/processed
#   make run             Launch Gradio app locally
#   make refresh         Rebuild embeddings/index from scratch
#   make test            Run tests (if tests/ exists)
#   make lint            Ruff lint (if installed)
#   make format          Black format (if installed)
#   make clean           Remove caches and build artifacts
#   make help            Show this help

# Detect platform and set venv bin path
ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
  PY_BOOT  ?= py -3.11
else
  VENV_BIN := .venv/bin
  PY_BOOT  ?= python3
endif

PY    := $(VENV_BIN)/python
PIP   := $(PY) -m pip
PYTEST:= $(VENV_BIN)/pytest
RUFF  := $(VENV_BIN)/ruff
BLACK := $(VENV_BIN)/black

APP            := app.py
RAW_DIR        := data/raw
PROCESSED_DIR  := data/processed
# Input dataset detection
INPUT_FMT   ?= jsonl
INPUT_FILE  ?= $(firstword $(wildcard $(RAW_DIR)/*.$(INPUT_FMT)))
INDEX_DIR      := $(PROCESSED_DIR)
EMBED_MODEL    := sentence-transformers/all-MiniLM-L6-v2
GEN_MODEL      := Qwen/Qwen2.5-0.5B-Instruct

# Fall back to requirements.txt only; dev reqs are optional
REQS_MAIN := requirements.txt
REQS_DEV  := requirements-dev.txt

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "BakeChat RAG — useful targets"
	@echo "  install        Create venv and install requirements"
	@echo "  build-index    Build FAISS index from $(RAW_DIR) into $(PROCESSED_DIR)"
	@echo "  run            Start Gradio app ($(APP))"
	@echo "  refresh        Delete $(INDEX_DIR) and rebuild"
	@echo "  test           Run pytest if tests/ exists"
	@echo "  lint           Ruff lint if available"
	@echo "  format         Black format if available"
	@echo "  clean          Remove caches and artifacts"
	@echo ""

# -----------------------------
# Setup
# -----------------------------
.PHONY: venv
venv: .venv/created

.venv/created:
	$(PY_BOOT) -m venv .venv
	@echo "Venv created at .venv"
	@# Upgrade pip/wheel/setuptools for sanity
	$(PIP) install --upgrade pip setuptools wheel
	@# Mark as created
	@echo "ok" > .venv/created

.PHONY: install
install: venv
	@test -f $(REQS_MAIN) && $(PIP) install -r $(REQS_MAIN) || echo "No $(REQS_MAIN); skipping."
	@# Optional dev dependencies
	@if [ -f $(REQS_DEV) ]; then \
		$(PIP) install -r $(REQS_DEV); \
	else \
		echo "No $(REQS_DEV); skipping dev deps."; \
	fi
	@echo "Install complete."

# -----------------------------
# Data & Index
# -----------------------------
# Build index. Your rag/build_index.py should accept CLI args like:
#   --in <raw_dir> --out <processed_dir> --model <embed_model>
.PHONY: build-index
build-index: venv
	@test -d $(RAW_DIR) || (echo "Missing $(RAW_DIR). Put your dataset there." && exit 1)
	@test -n "$(INPUT_FILE)" || (echo "No *.$(INPUT_FMT) found in $(RAW_DIR). Set INPUT_FILE=path or drop a file there." && exit 1)
	@mkdir -p $(PROCESSED_DIR)
	$(PY) -m rag.build_index --input "$(INPUT_FILE)" --format $(INPUT_FMT) --embedder "$(EMBED_MODEL)" --outdir "$(PROCESSED_DIR)"
	@echo "Index built at $(PROCESSED_DIR)."

.PHONY: refresh
refresh:
	@echo "Wiping index at $(INDEX_DIR)..."
	@rm -rf $(INDEX_DIR)
	@mkdir -p $(INDEX_DIR)
	$(MAKE) build-index

# -----------------------------
# Run App
# -----------------------------
.PHONY: run
run: venv
	# Use HF_HOME if you want a local cache dir: HF_HOME=.cache/huggingface make run
	$(PY) $(APP)

# -----------------------------
# Quality Gates (optional)
# -----------------------------
.PHONY: test
test: venv
	@if [ -d tests ]; then \
		$(PYTEST) -q || exit 1; \
	else \
		echo "No tests/ directory; skipping tests."; \
	fi

.PHONY: lint
lint: venv
	@if [ -x "$(RUFF)" ]; then \
		$(RUFF) check . ; \
	else \
		echo "Ruff not installed. Add it to $(REQS_DEV) to enable linting."; \
	fi

.PHONY: format
format: venv
	@if [ -x "$(BLACK)" ]; then \
		$(BLACK) . ; \
	else \
		echo "Black not installed. Add it to $(REQS_DEV) to enable formatting."; \
	fi

# -----------------------------
# Housekeeping
# -----------------------------
.PHONY: clean
clean:
	@echo "Cleaning caches and temporary junk..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .mypy_cache
	@echo "Done."
