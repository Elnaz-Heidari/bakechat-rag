import gradio as gr
from rag.query import RAG

rag = RAG(generator="Qwen/Qwen2.5-0.5B-Instruct")

def answer(q, k, use_reranker):
    docs = rag.retrieve(q, topk=int(k))
    if use_reranker and "rerank" not in docs.columns:
        from sentence_transformers import CrossEncoder
        rag.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        docs = rag.retrieve(q, topk=int(k))
    out = rag.generate(q, docs.head(5))
    table = docs.head(5)[["_id", "title", "ingredients"]]
    return out, table

with gr.Blocks(title="BakeChat RAG") as demo:
    gr.Markdown("# 🍞 BakeChat RAG — Recipes QA with Sources & Substitutions")
    with gr.Row():
        q = gr.Textbox(label="Ask a recipe question", placeholder="e.g., Egg-free substitute for brownies?")
    with gr.Row():
        k = gr.Slider(5, 50, value=20, step=1, label="Top-k retrieve")
        rr = gr.Checkbox(label="Use reranker (slower)")
    btn = gr.Button("Search & Answer")
    ans = gr.Markdown(label="Answer")
    tbl = gr.Dataframe(headers=["_id", "title", "ingredients"], label="Top sources", interactive=False)
    btn.click(fn=answer, inputs=[q, k, rr], outputs=[ans, tbl])

if __name__ == "__main__":
    demo.launch()
