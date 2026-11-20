"""Filter JSONL(.gz) shards using a fastText language-identification model."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ... import io

_model = None


def init_worker(model_path):
    import fasttext

    global _model
    _model = fasttext.load_model(str(model_path))


def process_file(task):
    source, target, language, probability = task
    total, retained = 0, 0
    with io.open_text(target, "wt") as stream:
        import json

        for record in io.read_jsonl(source):
            total += 1
            text = record.get("text", "").replace("\n", " ").strip()
            if not text:
                continue
            labels, probabilities = _model.predict(text)
            if labels and labels[0] == f"__label__{language}" and probabilities[0] >= probability:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                retained += 1
    return {"file": str(source), "total": total, "retained": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--input_dir", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--model-path", "--model_path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--min-probability", type=float, default=0.0)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.min_probability <= 1:
        parser.error("Require workers >= 1 and 0 <= min-probability <= 1")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ")
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    sources = sorted(
        p
        for p in args.input_dir.iterdir()
        if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
    )
    if not sources:
        raise ValueError(f"No JSONL(.gz) files under {args.input_dir}")
    tasks = [(p, args.output_dir / p.name, args.language, args.min_probability) for p in sources]
    if args.workers == 1:
        init_worker(args.model_path)
        results = list(map(process_file, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker, initargs=(args.model_path,)
        ) as pool:
            results = list(pool.map(process_file, tasks))
    io.write_json(args.output_dir / "language_filter_summary.json", results)
    print(
        f"retained {sum(r['retained'] for r in results)} / {sum(r['total'] for r in results)} documents"
    )


if __name__ == "__main__":
    main()
