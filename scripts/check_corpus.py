"""Optional CPU integration with synthetic OCR, DataTrove and fixed fastText weights."""

import gzip
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from cpt_memorization.io import read_jsonl, write_jsonl

root = Path(__file__).resolve().parents[1]
work = root / ".work/corpus-check"
work.mkdir(parents=True, exist_ok=True)
env = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "TMPDIR": str(root / ".work/tmp"),
    "XDG_CACHE_HOME": str(root / ".work/cache"),
}
texts = [
    "これは架空の文書です。図書館の開館時間について説明します。閲覧室は午前九時に開きます。" * 4,
    "これは架空の文書です。図書館の開館時間について説明します。閲覧室は午前九時に開きます。" * 4,
    "This synthetic English document describes a completely different topic about weather and clouds. "
    * 4,
]
with tarfile.open(work / "ocr.tar", "w") as archive:
    for index, text in enumerate(texts):
        xml = f'<OCRDATASET><PAGE><LINE TYPE="本文" STRING="{text}"/></PAGE></OCRDATASET>'
        content = gzip.compress(xml.encode())
        member = tarfile.TarInfo(f"synthetic-{index}.xml.gz")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))


def run(*args):
    subprocess.run([sys.executable, "-m", "cpt_memorization", *map(str, args)], env=env, check=True)


run("corpus-extract", work / "ocr.tar", work / "formatted.jsonl")
assert len(list(read_jsonl(work / "formatted.jsonl"))) == 3
run(
    "corpus-dedup",
    work / "formatted.jsonl",
    work / "dedup",
    "--no-slurm",
    "--workers",
    "1",
    "--force-clean",
)
shards = sorted((work / "dedup").glob("*/results/deduplicated_output/*.jsonl.gz"))
records = [record for shard in shards for record in read_jsonl(shard)]
assert len(records) == 2, (
    f"Expected two distinct publications after deduplication, got {len(records)}"
)

import fasttext
import numpy as np

training = work / "lid_training.txt"
training.write_text(
    (
        "__label__ja 日本語 図書館 開館 閲覧室 午前 九時\n"
        "__label__en English document weather clouds topic\n"
    )
    * 100,
    encoding="utf-8",
)
# Build only the dictionary, then install deterministic synthetic weights. The
# production path loads existing lid weights; training is not part of that path.
# This also avoids the upstream NaN seen when training this tiny toy dataset.
model = fasttext.train_supervised(
    str(training), epoch=0, dim=2, thread=1, minCount=1, verbose=0, bucket=0, minn=0, maxn=0
)
labels, words = model.get_labels(), model.get_words()
inputs = np.zeros((len(words), 2), dtype=np.float32)
outputs = np.eye(2, dtype=np.float32) * 8
for label, text in [
    ("__label__ja", "日本語 図書館 開館 閲覧室 午前 九時"),
    ("__label__en", "English document weather clouds topic"),
]:
    for word in text.split():
        inputs[model.get_word_id(word), labels.index(label)] = 1
model.set_matrices(inputs, outputs)
model.save_model(str(work / "toy-lid.bin"))
write_jsonl(
    work / "language-input/shard.jsonl.gz",
    [
        {"text": "日本語 図書館 開館 閲覧室 午前 九時", "id": "ja"},
        {"text": "English document weather clouds topic", "id": "en"},
    ],
)
run(
    "corpus-language",
    "--input-dir",
    work / "language-input",
    "--output-dir",
    work / "language-output",
    "--model-path",
    work / "toy-lid.bin",
    "--workers",
    "1",
)
kept = list(read_jsonl(work / "language-output/shard.jsonl.gz"))
assert [record["id"] for record in kept] == ["ja"]
print("Synthetic OCR → publication deduplication and gzip language filtering passed")
