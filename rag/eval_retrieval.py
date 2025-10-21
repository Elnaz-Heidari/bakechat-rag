import argparse, json, time, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Simple Recall@k + MRR evaluator over an eval file with fields:
# {"question": str, "relevant_ids": ["recipe_123", ...]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    eval_rows = [json.loads(l) for l in open(args.eval, "r", encoding="utf-8").read().splitlines()]
    meta = pd.read_parquet("data/processed/recipes.meta.parquet")
    id_list = meta["_id"].tolist()

    model = SentenceTransformer(args.embedder)
    index = faiss.read_index("data/processed/recipes.index.faiss")

    recalls = []
    mrrs = []
    latencies = []

    for ex in eval_rows:
        q = ex["question"]
        rel = set(ex["relevant_ids"]) if isinstance(ex["relevant_ids"], list) else set([ex["relevant_ids"]])
        t0 = time.time()
        qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        scores, idx = index.search(qv, args.topk)
        latencies.append(time.time() - t0)
        preds = [id_list[i] for i in idx[0]]

        # Recall@k
        hit = 1.0 if any(p in rel for p in preds) else 0.0
        recalls.append(hit)
        # MRR
        rr = 0.0
        for rank, p in enumerate(preds, 1):
            if p in rel:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    print({
        "Recall@%d" % args.topk: float(np.mean(recalls)),
        "MRR": float(np.mean(mrrs)),
        "avg_latency_s": float(np.mean(latencies)),
        "n": len(eval_rows)
    })

if __name__ == "__main__":
    main()
