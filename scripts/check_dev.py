"""Optional CPU integration for dev page/booklet MinHash, analysis and exact scans.

Run after installing corpus, analysis, tokenizers and plots extras. All data and
the tokenizer are synthetic; no embedding or language model is downloaded.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from cpt_memorization.corpus.stage import stage
from cpt_memorization.io import read_jsonl, write_json, write_jsonl

root = Path(__file__).resolve().parents[1]
(root / ".work/tmp").mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="dev-check-", dir=root / ".work/tmp"))
env = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "TMPDIR": str(root / ".work/tmp"),
    "XDG_CACHE_HOME": str(root / ".work/cache"),
    "HF_HOME": str(root / ".work/cache/huggingface"),
    "MPLCONFIGDIR": str(root / ".work/matplotlib"),
}


def run(*args):
    subprocess.run([sys.executable, "-m", "cpt_memorization", *map(str, args)], env=env, check=True)


def pages(pid, texts):
    return [
        {"text": text, "meta": {"pid": pid, "page_index": i, "page_count": len(texts)}}
        for i, text in enumerate(texts)
    ]


texts = [
    "これは架空の図書館の案内です。開館時間と閲覧室の利用方法について説明します。" * 4,
    "これは合成した気象の資料です。海洋の気温変化と低気圧の発達について記述しています。" * 4,
    "架空の植物研究の記録です。森林の樹木と土壌中の微生物の関係を調べています。" * 4,
]
other = [
    texts[0],
    "鉄道の運行表と駅舎の設計に関する架空の報告書です。安全な運輸を考察します。" * 4,
    "合成の数学の教材です。整数の因数分解や幾何学の証明を詳しく解説します。" * 4,
]
write_jsonl(work / "a/shard.jsonl", pages("A", texts) + pages("C", other))
write_jsonl(
    work / "b/shard.jsonl.gz",
    [{**r, "meta": {**r["meta"], "revision": 2}} for r in pages("B", texts)],
)
stage([work / "a", work / "b"], work / "staged")
run(
    "corpus-dedup-pages",
    work / "staged",
    work / "minhash",
    "--force-serial",
    "--audit-pages",
    "--write-deduplicated-output",
    "--booklet-shared-pages-threshold",
    "2",
)
result = work / "minhash/minhash-5gram-20buckets-10hashes/results"
decisions = [row for p in (result / "page_audit").glob("*.jsonl") for row in read_jsonl(p)]
assert len(decisions) == 9, decisions
assert sum(r["remove_flag"] for r in decisions) == 4, decisions
assert all(not r["remove_flag"] for r in decisions if r["pid"] == "B"), decisions
assert all(r["remove_flag"] for r in decisions if r["pid"] == "A"), decisions
assert sum(r["remove_flag"] for r in decisions if r["pid"] == "C") == 1
assert (result / "remove_ids_booklet").is_dir()
run(
    "corpus-audit-summary",
    "--audit-dir",
    result / "page_audit",
    "--output-jsonl",
    work / "pid-ratios.jsonl",
)
ratios = list(read_jsonl(work / "pid-ratios.jsonl"))
assert sorted(r["dup_ratio"] for r in ratios) == [0, 1 / 3, 1], ratios
run(
    "corpus-merge", "--inputs", result / "deduplicated_output", "--output", work / "merged.jsonl.gz"
)
assert len(list(read_jsonl(work / "merged.jsonl.gz"))) == 5

run(
    "corpus-analyze",
    "stats",
    "--path",
    work / "merged.jsonl.gz",
    "--text-key",
    "text",
    "--model",
    "characters",
    "--outdir",
    work / "stats",
)
stats = json.loads((work / "stats/stats.json").read_text())
assert stats["docs"] == 5 and stats["tokens_total"] == stats["chars_total"]
write_jsonl(
    work / "pii.jsonl",
    [{"text": "Synthetic contact: test@example.invalid"}, {"text": "No contact here"}],
)
run(
    "corpus-analyze",
    "pii",
    "--path",
    work / "pii.jsonl",
    "--text-key",
    "text",
    "--outdir",
    work / "pii",
)
assert any("email" in p.read_text() for p in (work / "pii").glob("*.csv"))

tokenizer_dir = work / "tokenizer"
tokenizer_dir.mkdir()
tokenizer = Tokenizer(models.WordLevel({"[BOS]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
write_json(tokenizer_dir / "tokenizer_config.json", {"bos_token": "[BOS]"})
write_json(tokenizer_dir / "special_tokens_map.json", {})
write_jsonl(work / "scan/corpus.jsonl.gz", [{"text": "a a a b"}, {"text": "a a"}])
rows = [
    {
        "suffix_token_ids": [2, 2],
        "predicted_token_ids": [2, 2],
        "suffix": "a a",
        "predicted": "a a",
        "snippet_length": 4,
    }
] * 2
write_jsonl(work / "completions.jsonl", rows)
run(
    "memory-count-completions",
    "--input-files",
    work / "completions.jsonl",
    "--processed-root",
    work / "scan",
    "--tokenizer",
    tokenizer_dir,
    "--workers",
    "1",
    "--batch-lines",
    "1",
    "--output-dir",
    work / "counts",
)
with (work / "counts/bleu_ge_0p75_samples_with_token_exact_count.csv").open() as stream:
    counts = list(csv.DictReader(stream))
assert [int(r["suffix_dup_count"]) for r in counts] == [3, 3], counts
from cpt_memorization.memorization.frequency.count_suffix_token_exact_matches import load_tokenizer

assert load_tokenizer(str(tokenizer_dir)).bos_token_id == 0

print(
    f"Dev integration passed: 9 pages → 5 retained (newer booklet wins), audited booklet decisions, stats, PII, exact counts. Artifacts: {work}"
)
