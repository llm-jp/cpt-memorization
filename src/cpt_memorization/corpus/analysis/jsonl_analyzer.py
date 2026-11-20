#!/usr/bin/env python3
import argparse, os, sys, json, math, random, gzip, io, re, statistics, itertools, pathlib
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import numpy as np

# Optional deps
try:
    import zstandard as zstd
except Exception:
    zstd = None


# Tokenizers
class TokenCounter:
    def __init__(self, model: str = "cl100k_base", hf_tokenizer: str = "gpt2"):
        if model == "characters":
            self.mode = "characters"
            self.count = len
            return
        self.mode = None
        self.model = model
        self.hf_tokenizer_name = hf_tokenizer
        try:
            import tiktoken

            self.enc = tiktoken.get_encoding(model)
            self.mode = "tiktoken"
        except Exception:
            try:
                from transformers import AutoTokenizer

                self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer)
                self.mode = "hf"
            except Exception:
                self.mode = "char"  # fallback
        if self.mode == "tiktoken":
            self.count = lambda s: len(self.enc.encode(s, disallowed_special=()))
        elif self.mode == "hf":
            self.count = lambda s: len(self.hf_tok.encode(s, add_special_tokens=False))
        else:
            self.count = lambda s: max(1, len(s) // 4)  # rough fallback


# I/O for big JSONL (supports .gz and .zst)
def open_maybe_compressed(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed. pip install zstandard")
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


# Dot-path extractor
def dig(obj: Dict[str, Any], path: str) -> Optional[Any]:
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def heuristic_text(obj: Dict[str, Any]) -> str:
    # Join all short string fields as a fallback
    parts = []

    def collect(o):
        if isinstance(o, str):
            parts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)

    collect(obj)
    # Limit size to avoid huge docs
    text = " ".join([p.strip() for p in parts if p and isinstance(p, str)])
    return text


# Reservoir sampler for uniform sampling from stream
class ReservoirSampler:
    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.n = 0
        self.buf = []
        self.rng = random.Random(seed)

    def add(self, item):
        self.n += 1
        if len(self.buf) < self.k:
            self.buf.append(item)
        else:
            j = self.rng.randint(1, self.n)
            if j <= self.k:
                self.buf[j - 1] = item

    def items(self):
        return self.buf


# -------- Stats --------
def run_stats(
    path: str,
    text_key: str,
    token_model: str,
    hf_tokenizer: str,
    hist_bins: int = 50,
    sample_for_pct: int = 100_000,
    id_key: str = None,
    max_docs: Optional[int] = None,
    skip_empty: bool = True,
    outdir: str = "out",
):
    import orjson
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    tc = TokenCounter(token_model, hf_tokenizer)
    count = 0
    char_sum = 0
    tok_sum = 0
    char_min, char_max = 10**18, 0
    tok_min, tok_max = 10**18, 0
    length_sample_chars = ReservoirSampler(sample_for_pct)
    length_sample_toks = ReservoirSampler(sample_for_pct)
    # For hist bins, we need online bin edges; use logarithmic bins over sample
    # We'll first collect lengths into small sample, then compute bins post-hoc using the sample itself.
    sampled_lengths_chars = []
    sampled_lengths_toks = []

    fh = open_maybe_compressed(path)
    for line in tqdm(fh, desc="stats-pass"):
        if max_docs and count >= max_docs:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
        except Exception:
            continue
        text = None
        if text_key:
            text = dig(obj, text_key)
            if not isinstance(text, str):
                text = None
        if text is None:
            text = heuristic_text(obj)
        if skip_empty and (text is None or text.strip() == ""):
            continue
        count += 1
        chars = len(text)
        toks = tc.count(text)
        char_sum += chars
        tok_sum += toks
        char_min = min(char_min, chars)
        char_max = max(char_max, chars)
        tok_min = min(tok_min, toks)
        tok_max = max(tok_max, toks)
        length_sample_chars.add(chars)
        length_sample_toks.add(toks)

    # Build histograms using sampled lengths
    sampled_lengths_chars = np.array(length_sample_chars.items(), dtype=np.int64)
    sampled_lengths_toks = np.array(length_sample_toks.items(), dtype=np.int64)

    def histo(sample, bins):
        if len(sample) == 0:
            edges = np.linspace(0, 1, bins + 1)
            counts = np.zeros(bins, dtype=np.int64)
            return edges, counts
        lo, hi = int(sample.min()), int(sample.max())
        # Use log-space bins if range is wide
        if hi > 0 and hi / max(1, lo) >= 100:
            edges = np.unique(
                np.round(np.logspace(math.log10(max(1, lo)), math.log10(hi), bins + 1))
            ).astype(int)
            if len(edges) < bins + 1:
                edges = np.linspace(lo, hi, bins + 1, dtype=int)
        else:
            edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1, dtype=int)
        counts, _ = np.histogram(sample, bins=edges)
        return edges, counts

    char_edges, char_counts = histo(sampled_lengths_chars, hist_bins)
    tok_edges, tok_counts = histo(sampled_lengths_toks, hist_bins)

    def pctiles(arr):
        if len(arr) == 0:
            return {}
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(arr, qs).tolist()
        return {f"p{q}": v for q, v in zip(qs, vals)}

    stats = {
        "docs": int(count),
        "chars_total": int(char_sum),
        "tokens_total": int(tok_sum),
        "chars_avg": float(char_sum / max(1, count)),
        "tokens_avg": float(tok_sum / max(1, count)),
        "chars_min": int(0 if count == 0 else char_min),
        "chars_max": int(char_max),
        "tokens_min": int(0 if count == 0 else tok_min),
        "tokens_max": int(tok_max),
        "char_percentiles_sampled": pctiles(sampled_lengths_chars),
        "token_percentiles_sampled": pctiles(sampled_lengths_toks),
        "tokenizer_mode": tc.mode,
    }
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Save histograms
    pd.DataFrame(
        {
            "char_left": char_edges[:-1],
            "char_right": char_edges[1:],
            "char_count": char_counts,
        }
    ).to_csv(os.path.join(outdir, "length_hist_chars.csv"), index=False)
    pd.DataFrame(
        {
            "tok_left": tok_edges[:-1],
            "tok_right": tok_edges[1:],
            "tok_count": tok_counts,
        }
    ).to_csv(os.path.join(outdir, "length_hist_tokens.csv"), index=False)

    print("Wrote:", os.path.join(outdir, "stats.json"))


# -------- Clustering (Arena §6.1-style) --------
def run_cluster(
    path: str,
    text_key: str,
    sample_size: int,
    st_model: str,
    device: str,
    min_cluster_size: int,
    outdir: str = "out",
):
    import orjson
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    # 1) Sample
    sampler = ReservoirSampler(sample_size)
    ids_sampler = ReservoirSampler(sample_size)
    fh = open_maybe_compressed(path)
    for i, line in enumerate(tqdm(fh, desc="sample-pass")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
        except Exception:
            continue
        text = dig(obj, text_key) if text_key else heuristic_text(obj)
        if not text or not isinstance(text, str) or not text.strip():
            continue
        doc_id = obj.get("id", obj.get("_id", i))
        sampler.add(text.strip())
        ids_sampler.add(doc_id)

    texts = sampler.items()
    if len(texts) == 0:
        print("No texts sampled. Check --text-key or input file.")
        return

    # 2) Embeddings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(st_model, device=device if device else None)
    # For E5-family, it's beneficial to prefix "passage: "
    if "e5" in st_model.lower():
        texts_to_encode = [f"passage: {t}" for t in texts]
    else:
        texts_to_encode = texts
    emb = model.encode(
        texts_to_encode, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    emb = np.asarray(emb)

    # 3) UMAP to 5D
    import umap

    reducer = umap.UMAP(n_components=5, metric="cosine", random_state=42, low_memory=True)
    X5 = reducer.fit_transform(emb)

    # 4) HDBSCAN
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(X5)
    n_clusters = int(len(set([l for l in labels if l != -1])))
    print(f"Found {n_clusters} clusters (+ noise).")

    # 5) Auto-label via c-TF-IDF keywords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import defaultdict

    cluster_docs = defaultdict(list)
    for t, l in zip(texts, labels):
        cluster_docs[l].append(t)
    # Build corpus per cluster (skip noise -1)
    cluster_texts = []
    cluster_ids = []
    for cid in sorted([c for c in cluster_docs.keys() if c != -1]):
        cluster_texts.append(" ".join(cluster_docs[cid]))
        cluster_ids.append(cid)
    if cluster_texts:
        vect = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
        tfidf = vect.fit_transform(cluster_texts)  # clusters x terms
        terms = np.array(vect.get_feature_names_out())
        top_terms = []
        for i in range(tfidf.shape[0]):
            row = tfidf.getrow(i).tocoo()
            idx = np.argsort(row.data)[-8:][::-1]
            top = [terms[j] for j in row.col[idx]]
            top_terms.append(", ".join(top))
    else:
        top_terms = []

    # 6) Summaries and examples
    rows = []
    rng = random.Random(0)
    example_records = []
    for cid in sorted([c for c in cluster_docs.keys() if c != -1]):
        docs = cluster_docs[cid]
        size = len(docs)
        # pick 5 random examples
        examples = rng.sample(docs, min(5, size))
        label = ""
        if cluster_ids:
            j = cluster_ids.index(cid)
            label = top_terms[j] if j < len(top_terms) else ""
        rows.append(
            {
                "cluster_id": int(cid),
                "size": int(size),
                "top_terms": label,
            }
        )
        for ex in examples:
            example_records.append(
                orjson.dumps({"cluster_id": int(cid), "text": ex}).decode("utf-8")
            )

    pd.DataFrame(rows, columns=["cluster_id", "size", "top_terms"]).sort_values(
        "size", ascending=False
    ).to_csv(os.path.join(outdir, "clusters.csv"), index=False)
    with open(os.path.join(outdir, "cluster_examples.jsonl"), "w", encoding="utf-8") as f:
        for line in example_records:
            f.write(line + "\n")
    print("Wrote:", os.path.join(outdir, "clusters.csv"))


# -------- PII detection --------
# Regex patterns (high recall)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.UNICODE)
URL_RE = re.compile(r"https?://[^\s]+", re.UNICODE)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b")
IPV6_RE = re.compile(r"\b([0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:0\d{1,4}|\d{2,4})[-\s]?\d{2,4}[-\s]?\d{3,4}")
JP_ZIP_RE = re.compile(r"(?:〒)?\d{3}-\d{4}")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
MYNUM_RE = re.compile(r"\b\d{12}\b")  # 日本の個人番号（12桁）—後でチェックディジット検証


def luhn_check(numstr: str) -> bool:
    s = [int(c) for c in re.sub(r"\D", "", numstr)]
    if len(s) < 12:
        return False
    total = 0
    parity = (len(s) - 2) % 2
    for i, d in enumerate(s[:-1]):
        if (i % 2) == parity:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return (total + s[-1]) % 10 == 0


def jp_mynum_check(numstr: str) -> bool:
    # JIS X 0301-like mod11,2 algorithm used for MyNumber check digit (simple variant)
    digits = [int(c) for c in numstr if c.isdigit()]
    if len(digits) != 12:
        return False
    s = 0
    for i, w in zip(digits[:11][::-1], itertools.cycle([2, 3, 4, 5, 6, 7])):  # common variant
        s += i * w
    cd = (11 - (s % 11)) % 10
    return cd == digits[-1]


def find_pii(text: str) -> List[str]:
    types = []
    if EMAIL_RE.search(text):
        types.append("email")
    if URL_RE.search(text):
        types.append("url")
    if IPV4_RE.search(text) or IPV6_RE.search(text):
        types.append("ip")
    if PHONE_RE.search(text):
        types.append("phone")
    if JP_ZIP_RE.search(text):
        types.append("jp_postal_code")
    # credit card with luhn
    for m in CC_RE.findall(text):
        if luhn_check(m):
            types.append("credit_card")
            break
    # mynumber
    for m in MYNUM_RE.findall(text):
        if jp_mynum_check(m):
            types.append("my_number")
            break
    return list(sorted(set(types)))


def run_pii(
    path: str,
    text_key: str,
    outdir: str = "out",
    spacy_ja: bool = False,
    spacy_en: bool = False,
    max_docs: Optional[int] = None,
):
    import orjson
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    n_flagged = 0
    rows = []
    n = 0
    if spacy_ja:
        try:
            import spacy

            nlp_ja = spacy.load("ja_core_news_sm")
        except Exception:
            nlp_ja = None
    else:
        nlp_ja = None
    if spacy_en:
        try:
            import spacy

            nlp_en = spacy.load("en_core_web_sm")
        except Exception:
            nlp_en = None
    else:
        nlp_en = None

    fh = open_maybe_compressed(path)
    for i, line in enumerate(tqdm(fh, desc="pii-pass")):
        if max_docs and n >= max_docs:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
        except Exception:
            continue
        text = dig(obj, text_key) if text_key else heuristic_text(obj)
        if not text or not isinstance(text, str):
            continue
        doc_id = obj.get("id", obj.get("_id", i))
        types = find_pii(text)
        # NER-based PERSON as optional
        if nlp_ja:
            try:
                for ent in nlp_ja(text[:2000]).ents:
                    if ent.label_ == "PERSON":
                        types.append("person_name_ja")
                        break
            except Exception:
                pass
        if nlp_en:
            try:
                for ent in nlp_en(text[:2000]).ents:
                    if ent.label_ == "PERSON":
                        types.append("person_name_en")
                        break
            except Exception:
                pass
        if types:
            n_flagged += 1
            excerpt = text[:200].replace("\n", " ")
            rows.append(
                {"doc_id": doc_id, "types": ",".join(sorted(set(types))), "sample_excerpt": excerpt}
            )
        n += 1

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "pii_flags.csv"), index=False)
    print(f"Flagged docs: {n_flagged} -> out/pii_flags.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "cluster", "pii"])
    ap.add_argument("--path", required=True, help="Path to .jsonl (.gz/.zst ok)")
    ap.add_argument(
        "--text-key", default="", help="Dot path to the main text field (default: heuristic)"
    )
    ap.add_argument("--outdir", default="out")
    # stats
    ap.add_argument("--model", default="cl100k_base", help="tiktoken encoding (stats)")
    ap.add_argument("--hf-tokenizer", default="gpt2", help="HF tokenizer fallback name")
    ap.add_argument("--hist-bins", type=int, default=50)
    ap.add_argument("--sample-for-pct", type=int, default=100_000)
    ap.add_argument("--max-docs", type=int, default=0, help="Optional cap for debug")
    # cluster
    ap.add_argument("--sample-size", type=int, default=50_000)
    ap.add_argument("--st-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--device", default="", help="'cuda' or ''")
    ap.add_argument("--min-cluster-size", type=int, default=32)
    # pii
    ap.add_argument("--spacy-ja", action="store_true")
    ap.add_argument("--spacy-en", action="store_true")

    args = ap.parse_args()
    max_docs = args.max_docs if args.max_docs > 0 else None

    if args.cmd == "stats":
        run_stats(
            args.path,
            args.text_key.strip() or "",
            args.model,
            args.hf_tokenizer,
            args.hist_bins,
            args.sample_for_pct,
            max_docs=max_docs,
            outdir=args.outdir,
        )
    elif args.cmd == "cluster":
        run_cluster(
            args.path,
            args.text_key.strip() or "",
            args.sample_size,
            args.st_model,
            args.device,
            args.min_cluster_size,
            outdir=args.outdir,
        )
    elif args.cmd == "pii":
        run_pii(
            args.path,
            args.text_key.strip() or "",
            outdir=args.outdir,
            spacy_ja=args.spacy_ja,
            spacy_en=args.spacy_en,
            max_docs=max_docs,
        )


if __name__ == "__main__":
    main()
