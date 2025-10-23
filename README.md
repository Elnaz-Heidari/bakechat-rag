🍞 BakeChat RAG — Retrieval-Augmented Recipe Assistant

BakeChat RAG is a mini Retrieval-Augmented Generation (RAG) application that answers recipe-related questions such as ingredient substitutions and preparation steps. It retrieves relevant recipes from a local dataset using FAISS similarity search, then uses a lightweight instruction-tuned LLM (Qwen 0.5B) to generate grounded responses with source citations. A Gradio UI provides an interactive chat-like interface.

---

🚀 Key Features

✅ FAISS-based semantic recipe retrieval  
✅ Detects substitution-type questions (e.g., “replace egg in brownies”)  
✅ Generates answers grounded in retrieved recipes  
✅ Uses Qwen/Qwen2.5-0.5B-Instruct (CPU-friendly LLM)  
✅ Gracefully falls back to summaries when unclear  
✅ Gradio UI for easy interaction  
✅ Easily extendable to large datasets (Kaggle, RecipeNLG)

---

🏗️ System Architecture

User Question → Embedding (MiniLM) → FAISS Retrieval → Context Builder → LLM (Qwen) → Answer + Sources → Diplay in UI


📉 (Diagram placeholder — add later as `assets/architecture.png`)

---

📂 Project Structure

bakechat-rag/
├─ README.md # Project documentation

├─ requirements.txt # Dependencies
├─ app.py # Gradio UI application
├─ Space.md # Notes for future Hugging Face deployment
├─ data/
│ ├─ raw/ # Raw recipe dataset (mini JSONL demo)
│ └─ processed/ # FAISS index + metadata parquet
├─ rag/
│ ├─ build_index.py # Creates FAISS index
│ ├─ query.py # Retrieval + generation logic
│ └─ eval_retrieval.py # Optional evaluation script
└─ prompts/
└─ system.txt # System prompt instructions


---

⚙️ How the RAG Pipeline Works

| Stage | What Happens |
|-------|--------------|
| 🔹 Embedding | Query converted to vector via MiniLM |
| 🔹 Retrieval | FAISS fetches top-matching recipes |
| 🔹 Context Build | Title, ingredients, steps merged |
| 🔹 Intent Check | Detects substitution questions |
| 🔹 Generation | Qwen LLM produces concise response |
| 🔹 Citation | Sources appended as `[Title | ID]` |

---

 ✅ Quickstart (Local Demo)

> Requires Python 3.10+ and PowerShell/macOS/Linux terminal.

1️⃣ Setup virtual environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Add a dataset

✅ Ready to run: A demo dataset (`mini_recipes.jsonl`) is already included in `data/raw/`, so you can run the app immediately.

📈 Optional: To scale up later, replace it with a larger dataset (e.g., Kaggle Food.com or RecipeNLG subset) and rebuild the FAISS index.

4️⃣ Build the FAISS index

python rag/build_index.py --input data/raw/mini_recipes.jsonl --format jsonl

5️⃣ Launch the Gradio app

python app.py

Then open: http://127.0.0.1:7860
💬 Example Queries
Query	What it does
“Replace eggs in brownies”	Returns bullet-style substitutions
“How to make vegan mayonnaise?”	Summarizes steps
“Gluten-free pizza dough method?”	Extracts instructions
“Alternative to milk in pancakes”	Suggests liquid replacements
🧠 Models Used
Component	Model
Embedder	sentence-transformers/all-MiniLM-L6-v2
Generator	Qwen/Qwen2.5-0.5B-Instruct
Optional Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2
⚠ **Note:** On the first run, the generator model (`Qwen/Qwen2.5-0.5B-Instruct`) will be downloaded automatically (~1 GB). Make sure you have a stable connection.

📊 Current Dataset

✅ Current: 5-item demo JSONL
📈 Ready for scaling using Kaggle/RecipeNLG datasets
✅ No pipeline changes required — just rebuild index.
📅 Roadmap
Status	Task
✅	Build working RAG demo
🔜	Add --limit support to indexing script
🔜	Scale to partial Kaggle dataset
🔜	Optional reranker toggle by default
🚀	Deploy to Hugging Face Spaces
👩‍💻 Author

Crafted by Elnaz as part of an LLM learning journey exploring real-world RAG architectures.
📄 License

MIT License – feel free to fork, learn, and extend!


---

✅ Go ahead and replace your existing README with this entire block.  
✅ Once complete, I’ll send you **Version A (detailed learning version)** for your own documentation and deeper understanding.

