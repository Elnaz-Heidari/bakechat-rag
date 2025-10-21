import argparse, os, time, json, pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# Minimal cleaning/normalization

def load_data(path: str, fmt: str, text_cols=None):
    if fmt == "jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                obj["_id"] = obj.get("id") or obj.get("recipe_id") or f"recipe_{i}"
                title = (obj.get("title") or "").strip()
                ingr = ", ".join(obj.get("ingredients", [])) if isinstance(obj.get("ingredients"), list) else str(obj.get("ingredients", ""))
                steps = "\n".join(obj.get("directions", [])) if isinstance(obj.get("directions"), list) else str(obj.get("steps", obj.get("instructions", "")))
                text = f"Title: {title}\nIngredients: {ingr}\nSteps: {steps}"
                rows.append({"_id": obj["_id"], "title": title, "ingredients": ingr, "steps": steps, "text": text})
        return pd.DataFrame(rows)
    elif fmt == "csv":
        df = pd.read_csv(path)
        text_cols = text_cols or ["title", "ingredients", "steps"]
        for col in text_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")
        df = df.fillna("")
        df["_id"] = df.get("id", pd.Series([f"recipe_{i}" for i in range(len(df))]))
        df["text"] = df[text_cols].astype(str).agg(lambda r: f"Title: {r[text_cols[0]]}\nIngredients: {r[text_cols[1]]}\nSteps: {r[text_cols[2]]}", axis=1)
        return df[["_id", *text_cols, "text"]].rename(columns={text_cols[0]:"title", text_cols[1]:"ingredients", text_cols[2]:"steps"})
    else:
        raise ValueError("Unsupported format: use jsonl or csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", choices=["jsonl","csv"], required=True)
    ap.add_argument("--text-cols", nargs="*", default=None, help="CSV: order is title ingredients steps")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("[load]", args.input)
    df = load_data(args.input, args.format, args.text_cols)

    print(f"[embed] model={args.embedder} n={len(df)}")
    model = SentenceTransformer(args.embedder)
    embs = model.encode(df["text"].tolist(), batch_size=128, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    faiss_path = os.path.join(args.outdir, "recipes.index.faiss")
    meta_path = os.path.join(args.outdir, "recipes.meta.parquet")

    print("[save]", faiss_path)
    faiss.write_index(index, faiss_path)
    print("[save]", meta_path)
    pd.DataFrame({"_id": df["_id"], "title": df["title"], "ingredients": df["ingredients"], "steps": df["steps"]}).to_parquet(meta_path, index=False)

    print("done")

if __name__ == "__main__":
    main()
