# 🍞 BakeChat RAG — Retrieval-Augmented Recipe Assistant

BakeChat RAG is a mini Retrieval-Augmented Generation (RAG) application that answers recipe-related questions such as ingredient substitutions and preparation steps. It retrieves relevant recipes from a local dataset using FAISS similarity search, then uses a lightweight instruction-tuned LLM (Qwen 0.5B) to generate grounded responses with source citations. A Gradio UI provides an interactive chat-like interface.

---

## 🚀 Key Features

- ✅ FAISS-based semantic recipe retrieval  
- ✅ Detects substitution-type questions (e.g., “replace egg in brownies”)  
- ✅ Generates answers grounded in retrieved recipes  
- ✅ Uses Qwen/Qwen2.5-0.5B-Instruct (CPU-friendly LLM)  
- ✅ Gracefully falls back to summaries when unclear  
- ✅ Gradio UI for easy interaction  
- ✅ Easily extendable to large datasets (Kaggle, RecipeNLG)

---

## 🏗️ System Architecture
```mermaid
flowchart LR
  A["User Question"] --> B["Embedder (MiniLM)"]
  B --> C["FAISS Index (Top-K Similar Recipes)"]
  C --> D["Context Builder (title, ingredients, steps)"]
  D --> E["LLM Generator (Qwen 0.5B)"]
  E --> F["Answer + Sources"]
  F --> G["Gradio UI"]
```



# 📂 Project Structure    
```text
bakechat-rag/
├─ README.md                 # This file
├─ requirements.txt          # Dependencies
├─ app.py                    # Gradio UI application
├─ Space.md                  # Notes for future Hugging Face deployment
├─ data/
│  ├─ raw/                   # Raw recipe dataset (mini JSONL demo)
│  └─ processed/             # FAISS index + metadata parquet
├─ rag/
│  ├─ build_index.py         # Creates FAISS index
│  ├─ query.py               # Retrieval + generation logic
│  └─ eval_retrieval.py      # Optional evaluation script
├─ prompts/
│  └─ system.txt             # System prompt instructions
└─ assets/
   └─ ui-screenshot.png      # Static UI screenshot (optional)
```



# ⚙️ How the RAG Pipeline Works
| Stage            | What Happens                         |
| ---------------- | ------------------------------------ |
| 🔹 Embedding     | Query converted to vector via MiniLM |
| 🔹 Retrieval     | FAISS fetches top-matching recipes   |
| 🔹 Context Build | Title, ingredients, steps merged     |
| 🔹 Intent Check  | Detects substitution questions       |
| 🔹 Generation    | Qwen LLM produces concise response   |
| 🔹 Citation      | Sources appended as `[Title \| ID]`  |


# 🎥 Quick Demo
assets/demo.jpg

✅ Quickstart (Local Demo)

Requires Python 3.10+ and PowerShell/macOS/Linux terminal.

1) Setup virtual environment
python -m venv .venv

Activate it:

- Windows (PowerShell): .\.venv\Scripts\activate

- macOS/Linux: source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt

3) Add a dataset

✅ Ready to run: A demo dataset (mini_recipes.jsonl) is already included in data/raw/, so you can run the app immediately.
📈 To scale up later, replace it with a larger dataset (e.g., Kaggle Food.com or a RecipeNLG subset) and rebuild the FAISS index.

4) Build the FAISS index
python rag/build_index.py --input data/raw/mini_recipes.jsonl --format jsonl

5) Launch the Gradio app
python app.py

Then open: http://127.0.0.1:7860


# ✅ First-Run Checklist
Python 3.10+ installed
Created and activated a virtual environment
Built the FAISS index with your dataset
Aware the model will download (~1 GB) the first time
Ran python app.py and opened the Gradio UI


# 💬 Example Queries
Query	Expected Behavior
“Replace eggs in brownies”	Bullet-style substitutions
“How to make vegan mayonnaise?”	Summarizes steps
“Gluten-free pizza dough method?”	Extracts instructions
“Alternative to milk in pancakes”	Suggests liquid replacements


# 🧠 Models Used
Component	Model
Embedder	sentence-transformers/all-MiniLM-L6-v2
Generator (default)	Qwen/Qwen2.5-0.5B-Instruct
Optional Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2


# 📊 Dataset

    ✅ Current: 5-item demo JSONL

    📈 Ready for scaling using Kaggle/RecipeNLG datasets

    💡 No pipeline changes required — just rebuild index after replacing the dataset.


# 📅 Roadmap
| Status | Task                                |
| ------ | ----------------------------------- |
| ✅     | Build working RAG demo              |
| 🔜     | Add `--limit` support to indexing   |
| 🔜     | Scale to partial Kaggle dataset     |
| 🔜     | Optional reranker toggle by default |
| 🚀     | Deploy to Hugging Face Spaces       |


# 👩‍💻 Author

Crafted by Elnaz as part of an LLM learning journey exploring real-world RAG architectures.


# 📄 License

MIT License – feel free to fork, learn, and extend!