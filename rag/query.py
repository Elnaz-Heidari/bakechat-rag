# rag/query.py
import argparse, re, pandas as pd
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

class RAG:
    def __init__(self, embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
                 reranker: Optional[str] = None, generator: str = "facebook/opt-350m"):
        # Retrieval bits
        self.embedder = SentenceTransformer(embedder)
        self.index = faiss.read_index("data/processed/recipes.index.faiss")
        self.meta = pd.read_parquet("data/processed/recipes.meta.parquet")
        self.reranker = CrossEncoder(reranker) if reranker else None
        # Generator bits
        cfg = AutoConfig.from_pretrained(generator)
        self.tok = AutoTokenizer.from_pretrained(generator)
        if self.tok.pad_token is None and hasattr(self.tok, "eos_token"):
            self.tok.pad_token = self.tok.eos_token
        if getattr(cfg, "is_encoder_decoder", False):
            self.gen = AutoModelForSeq2SeqLM.from_pretrained(generator, device_map="auto")
            self.is_encdec = True
        else:
            self.gen = AutoModelForCausalLM.from_pretrained(generator, device_map="auto")
            self.is_encdec = False

    def retrieve(self, query: str, topk: int = 20) -> pd.DataFrame:
        qv = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idx = self.index.search(qv, topk); idx = idx[0]
        cand = self.meta.iloc[idx].copy()
        cand["score_ip"] = scores[0]
        cand = cand.reset_index(drop=True)
        if self.reranker:
            pairs = [(query, f"{row.title}\n{row.ingredients}\n{row.steps}") for row in cand.itertuples(index=False)]
            rerank_scores = self.reranker.predict(pairs)
            cand["rerank"] = rerank_scores
            cand = cand.sort_values("rerank", ascending=False).reset_index(drop=True)
        return cand

    def generate(self, query: str, contexts: pd.DataFrame, max_new_tokens: int = 90) -> str:
        # ---- intent detection (substitution ONLY if clearly asked) ----
        q = (query or "").strip(); ql = q.lower()
        sub_trigs = [r"\breplace\b", r"\bsubstitute\b", r"\bswap\b",
                     r"\balternative(?:\s+to)?\b", r"\binstead of\b",
                     r"\bwithout\b", r"\ballergy to\b", r"\bcan['’]?t use\b", r"\bcant use\b"]
        is_sub = any(re.search(p, ql) for p in sub_trigs)
        # target extraction
        target = None
        for p in [r"(?:replace|substitute|swap)\s+(?:the\s+)?([a-zA-Z\- ]+?)\s+(?:in|for)\b",
                  r"(?:instead of|alternative(?:\s+to)?)\s+([a-zA-Z\- ]+)",
                  r"(?:without|allergy to|can['’]?t use|cant use)\s+([a-zA-Z\- ]+)",
                  r"(?:replace|substitute|swap)\s+(?:the\s+)?([a-zA-Z\- ]+)$"]:
            m = re.search(p, ql)
            if m: target = m.group(1).strip(); break
        if target:
            norm = target.replace("eggs","egg").replace("dairy milk","milk").strip()
            target = re.sub(r"\b(in|for|on|with)\b.*$", "", norm).strip()

        # ---- build compact context + citations ----
        recs = contexts[["_id","title","ingredients","steps"]].fillna("").to_dict("records")
        def rel_score(r):
            blob = f"{r['title']} {r['ingredients']} {r['steps']}".lower()
            toks = set(re.findall(r"[a-z]+", ql))
            score = sum(tok in blob for tok in toks)
            if target and target in blob: score += 2
            return score
        recs = sorted(recs, key=rel_score, reverse=True)
        blocks, cites = [], []
        for r in recs[:3]:
            title = r["title"].strip(); rid = str(r["_id"]).strip()
            ingr = str(r["ingredients"]).strip(); steps = str(r["steps"]).strip()
            blocks.append(f"[{title} | {rid}]\nIngredients: {ingr}\nSteps: {steps[:300]}")
            cites.append(f"[{title} | {rid}]")
        ctx = "\n\n".join(blocks)
        cite_line = "Sources: " + ", ".join(cites) if cites else "Sources: (none)"

        # ---- prompts ----
        base_sys = ("You are a grounded recipe assistant. Use ONLY the Context. "
                    "Cite sources as [Title | ID]. Write in third person. Avoid anecdotes.")
        if is_sub:
            if not target:
                return ("I couldn’t identify which ingredient you want to replace. "
                        "Please specify (e.g., “substitute for milk in pancakes?”).\n\n" + cite_line)
            sys = (base_sys + " Return 2–4 bullet points with safe substitutions for the requested ingredient, "
                              "each with a short reason and quantities if possible.")
            example = "Example:\n- 1/4 cup applesauce — adds moisture + binding.\n- 1 tbsp ground flax + 3 tbsp water — forms a gel (\"flax egg\").\n"
            prompt = f"{sys}\n\n{example}\nQuestion: {q}\n\nContext:\n{ctx}\n\nAnswer (bullets only):"
        else:
            sys = base_sys + " Keep the answer under 120 words. If steps exist, summarize them briefly."
            prompt = f"{sys}\n\nQuestion: {q}\n\nContext:\n{ctx}\n\nAnswer:"

        # ---- generate deterministically ----
        inputs = self.tok(prompt, return_tensors="pt").to(self.gen.device)
        if self.is_encdec:
            out = self.gen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.15)
            candidate = self.tok.decode(out[0], skip_special_tokens=True).strip()
        else:
            out = self.gen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    repetition_penalty=1.2, pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.tok.eos_token_id)
            gen_ids = out[0][inputs["input_ids"].shape[-1]:]
            candidate = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        def clean(s: str) -> str: return s.replace("\u2022","-").strip()
        cand = clean(candidate)

        if is_sub:
            lines = [l.strip() for l in cand.splitlines() if l.strip()]
            bullets = [l for l in lines if l.startswith("-") or l.startswith("•")]
            bad = (len(bullets) < 2 or "question:" in cand.lower() or "context:" in cand.lower())
            if not bad:
                return "\n".join(bullets[:4]) + "\n\n" + cite_line
            catalogs = {
              "egg": [("1/4 cup applesauce","adds moisture + binding"),
                      ("1/2 mashed ripe banana","binds; slight banana taste"),
                      ("3 tbsp aquafaba","whipped chickpea brine binds/aerates"),
                      ("1 tbsp ground flax + 3 tbsp water","forms a gel (\"flax egg\")")],
              "milk":[("same amount oat/soy/almond milk","similar liquid content"),
                      ("3 tbsp yogurt + splash water","adds moisture + mild tang"),
                      ("water + 1 tsp oil per 1/2 cup","keeps batter fluid")],
              "butter":[("equal oil","neutral moisture/fat; crumb slightly different"),
                        ("margarine","similar fat; check salt"),
                        ("coconut oil","adds subtle coconut flavor")],
              "mayonnaise":[("3 tbsp aquafaba + 1 tsp mustard + 1 tsp lemon","emulsifies like mayo"),
                            ("plain yogurt","tangy binder; adjust salt"),
                            ("silken tofu (blended)","neutral creamy binder")],
            }
            key = None
            for k in catalogs:
                if re.search(rf"\b{k}s?\b", (target or "")): key = k; break
            if key is None:
                for k in catalogs:
                    if re.search(rf"\b{k}s?\b", ql): key = k; break
            if key:
                items = catalogs[key][:4]
                return "\n".join([f"- {amt} — {why}." for amt, why in items]) + "\n\n" + cite_line
            return ("I need the ingredient to replace (e.g., egg, milk, butter). Please rephrase your question.\n\n" + cite_line)

        # General QA guardrails
        off_topic = any(bad_kw in cand.lower() for bad_kw in ["pizza"," i ","i'","i’m","i am","my ","when i"])
        too_short = len(cand) < 30; echoes = ("question:" in cand.lower() or "context:" in cand.lower())
        if not (off_topic or too_short or echoes):
            return cand + "\n\n" + cite_line
        if recs:
            r0 = recs[0]; title = r0["title"].strip()
            ing = str(r0["ingredients"]).strip()
            stp = re.sub(r"\s{2,}"," ", str(r0["steps"]).strip().replace("\n"," "))
            summary = f"{title}: Ingredients include {ing}. Method (brief): {stp[:280].rstrip('.')}.";
            return summary + "\n\n" + cite_line
        return "I couldn't find enough context to answer confidently.\n\n" + cite_line

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--generator", default="facebook/opt-350m")
    args = ap.parse_args()
    rag = RAG(reranker=args.reranker, generator=args.generator)
    docs = rag.retrieve(args.query, topk=args.topk)
    print("Top docs:\n", docs.head(5)[["_id","title","score_ip"]])
    print("\nAnswer:\n", rag.generate(args.query, docs.head(5)))

if __name__ == "__main__":
    main()
