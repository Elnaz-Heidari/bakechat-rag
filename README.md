# 🍞 BakeChat RAG — A Retrieval-Augmented Recipe Assistant

BakeChat RAG is an educational-to-professional learning project that demonstrates how to build a real Retrieval-Augmented Generation (RAG) app from scratch — using FAISS-based document retrieval and a Qwen instruction-tuned LLM to answer cooking questions with cited recipe sources.

---

## 📌 Project Overview

This project simulates a mini recipe assistant capable of:
- Retrieving relevant recipes from a local dataset
- Understanding ingredient substitution requests (e.g., “What can I use instead of eggs?”)
- Generating grounded answers using retrieved context + an LLM
- Citing original recipe sources in each response
- Running locally with a user-friendly Gradio interface

The goal is to learn the full RAG pipeline end-to-end:
embedding creation → FAISS indexing → semantic retrieval → prompt design → LLM generation → UI.

---

## 🎯 What This Project Teaches

| Concept | Why It Matters |
|--------|----------------|
| Embeddings | Convert text into numeric meaning for retrieval |
| FAISS similarity search | Enables scalable, fast document retrieval |
| Retrieval-Augmented Generation | Grounds LLMs on real data (reduce hallucinations) |
| Prompt engineering | Controls LLM behavior (bullets vs summary) |
| Generator model loading (Qwen/OPT/T5) | Swap models without changing pipeline |
| Gradio UI | Turn your logic into a usable app |

---

## 🏗️ System Architecture

1. **User asks a question** (e.g., “How can I replace eggs in brownies?”)  
2. **Embed query** with SentenceTransformer (`all-MiniLM-L6-v2`)  
3. **FAISS** retrieves top-N similar recipes  
4. **Context builder** combines title + ingredients + steps  
5. **LLM (Qwen 0.5B Instruct by default)** generates an answer from context  
6. **Answer is formatted + cited** (e.g., `[Title | ID]`)  
7. **Displayed in Gradio** (web UI)

📉 Architecture Diagram:  
![Architecture Diagram](assets/architecture.png)

---

## 📂 Project Structure

