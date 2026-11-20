"""Check parallel generation file contracts and source/completion row identity."""

from pathlib import Path
import tempfile
import unittest

from cpt_memorization.io import read_jsonl, write_jsonl
from cpt_memorization.memorization.pipeline.shards import (
    merge_completions,
    shard_path,
    split_samples,
)
from cpt_memorization.memorization.analysis.base_exclusion_split import (
    make_output_row,
    split_rows,
)


class MemoryShardsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = self.work / "samples.jsonl"
        self.rows = [
            dict(
                doc_id=f"doc-{i // 5}",
                row_index=i // 5,
                snippet_length=length,
                prefix_token_ids=[i] * (length - 2),
                suffix_token_ids=[65, 66],
            )
            for i, length in enumerate([4, 6, 8, 10, 12] * 3)
        ]
        write_jsonl(self.source, self.rows)

    def completed_parts(self, workers):
        directory = self.work / f"samples-{workers}"
        counts = split_samples(self.source, directory, workers)
        self.assertEqual(sum(counts), len(self.rows))
        self.assertLessEqual(max(counts) - min(counts), 1)
        paths = []
        for index in range(workers):
            path = self.work / f"completion-{workers}-{index}.jsonl"
            rows = list(read_jsonl(shard_path(directory, index)))
            for row in rows:
                row.update(predicted_token_ids=[65, 66], predicted="AB", generation={"seed": 42})
            write_jsonl(path, rows)
            paths.append(path)
        return paths

    def test_parallel_results_restore_exact_input_order_for_uneven_shards(self):
        for workers in (1, 2, 4, 8):
            with self.subTest(workers=workers):
                parts = self.completed_parts(workers)
                output = self.work / f"merged-{workers}.jsonl"
                self.assertEqual(merge_completions(self.source, parts, output), 15)
                actual = list(read_jsonl(output))
                self.assertEqual([{k: row[k] for k in self.rows[0]} for row in actual], self.rows)

    def test_invalid_shards_never_publish_output(self):
        for defect in ("missing", "extra", "reordered", "short", "wrong-tokens", "absent"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as subdir:
                parts = self.completed_parts_for_subdir(Path(subdir))
                rows = list(read_jsonl(parts[0]))
                if defect == "missing":
                    rows.pop()
                elif defect == "extra":
                    rows.append(rows[-1])
                elif defect == "reordered":
                    rows.reverse()
                elif defect == "short":
                    rows[0]["predicted_token_ids"] = [65]
                elif defect == "wrong-tokens":
                    rows[0]["prefix_token_ids"][0] = 999
                if defect == "absent":
                    parts[0].unlink()
                else:
                    write_jsonl(parts[0], rows)
                output = Path(subdir) / "merged.jsonl"
                with self.assertRaises((ValueError, FileNotFoundError)):
                    merge_completions(self.source, parts, output)
                self.assertFalse(output.exists())
                self.assertEqual(list(output.parent.glob(".merged.jsonl.*.tmp")), [])

    def completed_parts_for_subdir(self, directory):
        parts = []
        for index in range(4):
            path = directory / f"{index}.jsonl"
            write_jsonl(
                path,
                [
                    dict(row, predicted_token_ids=[65, 66], predicted="AB")
                    for row in self.rows[index::4]
                ],
            )
            parts.append(path)
        return parts

    def test_split_rejects_empty_shards_and_existing_destination(self):
        target = self.work / "parts"
        with self.assertRaises(ValueError):
            split_samples(self.source, target, 16)
        self.assertFalse(target.exists())
        split_samples(self.source, target, 2)
        with self.assertRaises(FileExistsError):
            split_samples(self.source, target, 2)

    def test_merge_preserves_existing_output(self):
        output = self.work / "merged.jsonl"
        output.write_text("existing\n")
        with self.assertRaises(FileExistsError):
            merge_completions(self.source, [self.source], output)
        self.assertEqual(output.read_text(), "existing\n")

    def test_base_exclusion_separates_prefix_lengths_with_shared_corpus_row(self):
        target = [make_output_row(i, row, 1.0, "target") for i, row in enumerate(self.rows)]
        base = [
            make_output_row(i, row, 1.0, "base") for i, row in enumerate(self.rows) if i % 2 == 0
        ]
        self.assertEqual([row["row_index"] for row in target], list(range(15)))
        self.assertEqual([row["source_row_index"] for row in target], [i // 5 for i in range(15)])
        kept, overlap, mismatches = split_rows(target, base)
        self.assertEqual([row["row_index"] for row in overlap], list(range(0, 15, 2)))
        self.assertEqual([row["row_index"] for row in kept], list(range(1, 15, 2)))
        self.assertEqual(mismatches, [])
        original = {"archive_member": "old-document", "snippet_length": 1000}
        legacy = make_output_row(7, original, 0.8, "legacy")
        self.assertEqual(legacy["row_index"], 7)
        self.assertNotIn("source_row_index", legacy)


if __name__ == "__main__":
    unittest.main()
