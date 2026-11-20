import json
import argparse
from pathlib import Path
import gzip


def iter_input_paths(in_dir: Path):
    paths = sorted(in_dir.glob("*.jsonl.gz")) + sorted(in_dir.glob("*.jsonl"))
    for p in paths:
        yield p


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="除去されるページ（最初と最後）のサンプルを表示する"
    )
    parser.add_argument("--in-dir", type=Path, required=True, help="入力ディレクトリ")
    parser.add_argument(
        "--sample-count", type=int, default=5, help="表示するサンプル数（冊子単位）"
    )
    args = parser.parse_args()

    count = 0
    for path in iter_input_paths(args.in_dir):
        if count >= args.sample_count:
            break

        with open_text(path) as f:
            lines = f.readlines()

        # ページ単位で処理されているため、メタデータを見て最初と最後を探す
        first_page = None
        last_page = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                meta = obj.get("meta", {})
                page_index = meta.get("page_index")
                page_count = meta.get("page_count")

                if page_index is not None and page_count is not None:
                    if page_index == 0:
                        first_page = obj
                    elif page_index == page_count - 1:
                        last_page = obj
            except:
                continue

        if first_page or last_page:
            count += 1
            # そのまま（Raw JSON）を表示
            if first_page:
                print(json.dumps(first_page, ensure_ascii=False))
            if last_page:
                print(json.dumps(last_page, ensure_ascii=False))


if __name__ == "__main__":
    main()
