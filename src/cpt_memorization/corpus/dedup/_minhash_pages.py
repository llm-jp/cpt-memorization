"""
Unified MinHash dedup pipeline with optional audit logging and Slurm execution.
Based on:
https://github.com/llm-jp/scripts/blob/main/corpus/llm-jp-corpus-v4/common/dedup/minhash/slurm/minhash_dedup.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import tempfile
import time
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Iterable


def load_datatrove_dependencies():
    from datatrove.io import get_datafolder
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor
    from datatrove.pipeline.base import PipelineStep
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.logging import logger
    from datatrove.utils.stats import PipelineStats
    from datatrove.utils.typeshelper import Languages

    return {
        "LocalPipelineExecutor": LocalPipelineExecutor,
        "SlurmPipelineExecutor": SlurmPipelineExecutor,
        "PipelineStep": PipelineStep,
        "get_datafolder": get_datafolder,
        "MinhashDedupSignature": MinhashDedupSignature,
        "MinhashConfig": MinhashConfig,
        "MinhashDedupBuckets": MinhashDedupBuckets,
        "MinhashDedupCluster": MinhashDedupCluster,
        "MinhashDedupFilter": MinhashDedupFilter,
        "JsonlReader": JsonlReader,
        "JsonlWriter": JsonlWriter,
        "HashConfig": HashConfig,
        "logger": logger,
        "PipelineStats": PipelineStats,
        "Languages": Languages,
    }


_deps = None
PageAuditStep = object
PageDecisionWriter = object


def deps():
    global _deps
    if _deps is None:
        _deps = load_datatrove_dependencies()
    return _deps


class SerialLocalPipelineExecutor(deps()["LocalPipelineExecutor"]):
    """Fallback local executor that avoids multiprocessing.Manager."""

    def run(self):
        logger = deps()["logger"]
        PipelineStats = deps()["PipelineStats"]

        if self.depends:
            if not self.depends._launched:
                logger.info('Launching dependency job "{}"', self.depends)
                self.depends.run()
            while (incomplete := len(self.depends.get_incomplete_ranks())) > 0:
                logger.info(
                    "Dependency job still has {}/{} tasks. Waiting...",
                    incomplete,
                    self.depends.world_size,
                )
                time.sleep(10)

        self._launched = True
        ranks_range = range(self.local_rank_offset, self.local_rank_offset + self.local_tasks)
        if all(map(self.is_rank_completed, ranks_range)):
            logger.info(
                "Not doing anything as all {} tasks have already been completed.", self.local_tasks
            )
            return

        if self.workers != 1:
            logger.warning("SerialLocalPipelineExecutor forcing workers=1.")
            self.workers = 1

        self.save_executor_as_json()

        ranks_to_run = self.get_incomplete_ranks(ranks_range)
        if (skipped := self.local_tasks - len(ranks_to_run)) > 0:
            logger.info("Skipping {} already completed tasks", skipped)

        stats_results = []
        original_pipeline = self.pipeline
        for rank in ranks_to_run:
            self.pipeline = deepcopy(original_pipeline)
            logger.info(
                "Starting local task rank={} ({}/{})",
                rank,
                len(stats_results) + 1,
                len(ranks_to_run),
            )
            started_at = time.time()
            try:
                stats_results.append(self._run_for_rank(rank, local_rank=0))
            except Exception:
                logger.exception(
                    "Local task rank={} failed after {:.2f}s. logging_dir={}",
                    rank,
                    time.time() - started_at,
                    self.logging_dir,
                )
                raise
            logger.info(
                "Finished local task rank={} in {:.2f}s",
                rank,
                time.time() - started_at,
            )
        self.pipeline = original_pipeline

        aggregated_stats = sum(stats_results, start=PipelineStats())
        with self.logging_dir.open("stats.json", "wt") as statsfile:
            aggregated_stats.save_to_disk(statsfile)
        logger.success(aggregated_stats.get_repr(f"All {self.local_tasks} tasks"))
        return aggregated_stats


class PageAuditLogger:
    """Persist page-level removal decisions for downstream QA and PID aggregation."""

    def __init__(self, output_dir: str, filter_name: str = "minhash_page_dedup"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[int, object] = {}
        self.filter_name = filter_name

    def _get_handle(self, rank: int):
        if rank not in self._handles:
            filepath = self.output_dir / f"audit_{rank:05d}.jsonl"
            self._handles[rank] = open(filepath, "a", encoding="utf-8")
        return self._handles[rank]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handles = {}

    def _extract_metadata(self, document) -> dict:
        if hasattr(document, "metadata"):
            base_meta = document.metadata or {}
        else:
            base_meta = document.get("meta", {}) if isinstance(document, dict) else {}

        if isinstance(base_meta, dict) and isinstance(base_meta.get("meta"), dict):
            nested = base_meta.get("meta") or {}
            base_meta = {
                **{key: value for key, value in base_meta.items() if key != "meta"},
                **nested,
            }

        return base_meta

    def log(self, document, rank: int, decision: str, shard_id: int | None = None):
        meta = self._extract_metadata(document)
        page_count = meta.get("page_count")
        if page_count is None and isinstance(meta.get("pages"), list):
            page_count = len(meta["pages"])
        page_id = getattr(document, "id", None) or meta.get("page_id")
        if page_id is None:
            pid = meta.get("pid")
            page_index = meta.get("page_index")
            if pid is not None and page_index is not None:
                page_id = f"{pid}:{page_index}"

        payload = {
            "filter_name": self.filter_name,
            "page_id": page_id,
            "pid": meta.get("pid"),
            "page_index": meta.get("page_index"),
            "page_count": page_count,
            "booklet_id": meta.get("pid") or meta.get("archive_member") or meta.get("file_path"),
            "remove_flag": decision == "removed",
            "rank": rank,
        }
        if shard_id is not None:
            payload["shard_id"] = shard_id
        if "minhash_cluster_id" in meta:
            payload["cluster_id"] = meta.get("minhash_cluster_id")
        if "minhash_cluster_size" in meta:
            payload["cluster_size"] = meta.get("minhash_cluster_size")

        handle = self._get_handle(rank)
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"))
    )


def collect_stage3_source_files(input_path: Path, all_files: list[Path]) -> list[str]:
    """Preserve logical paths under the provided input root for stage 3.5 reloads.

    When the input directory is a staging tree of symlinks, resolving paths here would
    drop the staging-relative parent directories that stage 3.5 needs to reopen files.
    """
    if input_path.is_file():
        return [str(input_path)]
    return [str(path.relative_to(input_path)) for path in all_files]


class UnionFind:
    def __init__(self):
        self.parent_map: dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self.parent_map.get(item, item)
        if parent != item:
            parent = self.find(parent)
            self.parent_map[item] = parent
            return parent
        self.parent_map[item] = item
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent_map[root_right] = root_left


def _extract_metadata_from_raw_record(row: dict) -> dict:
    metadata: dict = {}
    top_level_meta = row.get("meta")
    if isinstance(top_level_meta, dict):
        metadata = dict(top_level_meta)

    raw_metadata = row.get("metadata", {})
    if isinstance(raw_metadata, dict):
        metadata = {**raw_metadata, **metadata}
    elif raw_metadata not in (None, {}):
        metadata = {**metadata, "metadata": raw_metadata}

    if isinstance(metadata.get("meta"), dict):
        nested = metadata.get("meta") or {}
        metadata = {**{key: value for key, value in metadata.items() if key != "meta"}, **nested}
    remaining = {
        key: value for key, value in row.items() if key not in {"text", "id", "media", "metadata"}
    }
    return metadata | remaining


def _open_jsonl(data_folder, filepath: str):
    path_obj = Path(filepath)
    if path_obj.is_absolute() and path_obj.exists():
        import fsspec

        return fsspec.open(str(path_obj), "rt", compression="infer").open()
    return data_folder.open(filepath, "r", compression="infer")


def _iter_binary_pairs(path: Path) -> Iterable[tuple[int, int]]:
    if not path.exists():
        return ()
    data = path.read_bytes()
    if len(data) % 8 != 0:
        raise ValueError(f"Invalid pair file length: {path}")
    return (struct.unpack("<II", data[idx : idx + 8]) for idx in range(0, len(data), 8))


def _write_binary_uints(path: Path, values: Iterable[int]) -> None:
    with path.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<I", value))


def _parse_revision_score(meta: dict) -> int:
    fields = [
        meta.get("edition"),
        meta.get("version"),
        meta.get("revision"),
        meta.get("title"),
        meta.get("book_title"),
    ]
    best = 0
    for value in fields:
        if not value:
            continue
        text = str(value)
        if "改訂" in text:
            best = max(best, 1)
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            try:
                best = max(best, int(digits))
            except ValueError:
                pass
    return best


def _parse_date_score(meta: dict) -> int:
    for key in (
        "publication_date",
        "published_at",
        "issue_date",
        "revised_at",
        "date",
        "year",
    ):
        value = meta.get(key)
        if value is None:
            continue
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) >= 4:
            try:
                return int(digits[:8])
            except ValueError:
                continue
    return 0


class RewriteRemoveIdsByBooklet(deps()["PipelineStep"]):
    type = "🫂 - DEDUP"
    name = "🎯 MinHash stage 3.5"
    _pair_record_format = "<III"
    _pair_partition_count = 64

    def __init__(
        self,
        *,
        source_input_path: str,
        source_files: list[str],
        source_remove_folder: str,
        output_folder: str,
        total_tasks: int,
        shared_pages_threshold: int,
        containment_threshold: float,
    ):
        super().__init__()
        self.source_input_path = source_input_path
        self.source_files = source_files
        self.source_remove_folder = Path(source_remove_folder)
        self.output_folder = Path(output_folder)
        self.total_tasks = total_tasks
        self.shared_pages_threshold = shared_pages_threshold
        self.containment_threshold = containment_threshold

    @classmethod
    def _pair_record(cls) -> struct.Struct:
        return struct.Struct(cls._pair_record_format)

    @staticmethod
    def _booklet_id(page_id: str | None, meta: dict) -> str:
        for key in ("pid", "booklet_id", "archive_member", "source_file", "file_path"):
            value = meta.get(key)
            if value not in (None, ""):
                return str(value)
        return page_id or "unknown"

    def _collect_cluster_members(self):
        cluster_members: dict[int, list[tuple[int, int]]] = {}
        needed_doc_idxs: dict[int, set[int]] = {
            task_rank: set() for task_rank in range(self.total_tasks)
        }
        skipped_cluster_paths = 0

        for cluster_path in sorted(self.source_remove_folder.glob("*.clusters")):
            file_rank = int(cluster_path.stem)
            if file_rank >= self.total_tasks:
                skipped_cluster_paths += 1
                continue
            for doc_idx, cluster_id in _iter_binary_pairs(cluster_path):
                doc_ref = (file_rank, doc_idx)
                cluster_members.setdefault(cluster_id, []).append(doc_ref)
                needed_doc_idxs[file_rank].add(doc_idx)

        return cluster_members, needed_doc_idxs, skipped_cluster_paths

    def _load_documents(self, needed_doc_idxs: dict[int, set[int]] | None = None):
        data_folder = deps()["get_datafolder"](self.source_input_path)
        doc_to_booklet: dict[tuple[int, int], str] = {}
        booklet_docs: dict[str, list[tuple[int, int]]] = {}
        booklet_meta: dict[str, dict] = {}
        input_root = Path(self.source_input_path).expanduser().resolve()

        for rank, source_file in enumerate(self.source_files):
            source_path = Path(source_file)
            if not source_path.is_absolute():
                open_path = str(source_path)
            elif input_root.is_dir():
                try:
                    open_path = str(source_path.relative_to(input_root))
                except ValueError:
                    open_path = source_path.name
            else:
                open_path = str(source_path)
            wanted_doc_idxs = None if needed_doc_idxs is None else needed_doc_idxs.get(rank, set())
            if wanted_doc_idxs is not None and not wanted_doc_idxs:
                continue
            doc_idx = 0
            with _open_jsonl(data_folder, open_path) as handle:
                for line_no, line in enumerate(handle):
                    if wanted_doc_idxs is not None and doc_idx not in wanted_doc_idxs:
                        doc_idx += 1
                        continue
                    row = json.loads(line)
                    meta = _extract_metadata_from_raw_record(row)
                    page_id = row.get("id", f"{open_path}/{line_no}")
                    booklet_id = self._booklet_id(page_id, meta)
                    doc_ref = (rank, doc_idx)
                    doc_to_booklet[doc_ref] = booklet_id
                    booklet_docs.setdefault(booklet_id, []).append(doc_ref)
                    booklet_meta.setdefault(booklet_id, meta)
                    doc_idx += 1
        return doc_to_booklet, booklet_docs, booklet_meta

    def _winner_key(
        self,
        booklet_idx: int,
        booklet_ids: list[str],
        booklet_docs: dict[str, list[tuple[int, int]]],
        booklet_meta: dict[str, dict],
    ):
        booklet_id = booklet_ids[booklet_idx]
        meta = booklet_meta.get(booklet_id, {})
        return (
            _parse_revision_score(meta),
            _parse_date_score(meta),
            len(booklet_docs.get(booklet_id, [])),
            -booklet_idx,
        )

    def _iter_shared_page_overlaps(
        self,
        *,
        cluster_members: dict[int, list[tuple[int, int]]],
        doc_to_booklet: dict[tuple[int, int], str],
        booklet_idx_by_id: dict[str, int],
        booklet_sizes: list[int],
        logger,
    ) -> Iterable[tuple[int, int, int]]:
        pair_count = 0
        skipped_cluster_refs = 0
        partition_count = self._pair_partition_count

        with tempfile.TemporaryDirectory(
            dir=self.output_folder,
            prefix=".shared_pages_",
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            partition_paths = [temp_dir / f"{idx:03d}.bin" for idx in range(partition_count)]
            partition_handles = [path.open("ab") for path in partition_paths]
            try:
                for members in cluster_members.values():
                    booklet_counts: dict[int, int] = {}
                    for doc_ref in members:
                        booklet_id = doc_to_booklet.get(doc_ref)
                        if booklet_id is None:
                            skipped_cluster_refs += 1
                            continue
                        booklet_idx = booklet_idx_by_id[booklet_id]
                        booklet_counts[booklet_idx] = booklet_counts.get(booklet_idx, 0) + 1
                    cluster_booklet_idxs = sorted(booklet_counts)
                    if len(cluster_booklet_idxs) < 2:
                        continue
                    for left_pos, left_idx in enumerate(cluster_booklet_idxs):
                        left_count = booklet_counts[left_idx]
                        for right_idx in cluster_booklet_idxs[left_pos + 1 :]:
                            overlap = min(left_count, booklet_counts[right_idx])
                            partition = ((left_idx * 1315423911) ^ right_idx) % partition_count
                            partition_handles[partition].write(
                                self._pair_record().pack(left_idx, right_idx, overlap)
                            )
                            pair_count += 1
            finally:
                for handle in partition_handles:
                    handle.close()

            logger.info(
                "Spilled {} booklet pair overlaps across {} partitions. skipped_cluster_refs={}",
                pair_count,
                partition_count,
                skipped_cluster_refs,
            )

            for partition_idx, partition_path in enumerate(partition_paths):
                if not partition_path.exists():
                    continue
                data = partition_path.read_bytes()
                if not data:
                    continue
                pair_record = self._pair_record()
                if len(data) % pair_record.size != 0:
                    raise ValueError(f"Invalid shared-pair spill file length: {partition_path}")
                aggregated: dict[tuple[int, int], int] = {}
                for offset in range(0, len(data), pair_record.size):
                    left_idx, right_idx, overlap = pair_record.unpack(
                        data[offset : offset + pair_record.size]
                    )
                    pair = (left_idx, right_idx)
                    aggregated[pair] = aggregated.get(pair, 0) + overlap
                logger.info(
                    "Aggregated {} booklet pairs from spill partition {}/{}.",
                    len(aggregated),
                    partition_idx + 1,
                    partition_count,
                )
                for (left_idx, right_idx), overlap in aggregated.items():
                    denominator = min(booklet_sizes[left_idx], booklet_sizes[right_idx])
                    if denominator == 0:
                        continue
                    containment = overlap / denominator
                    if (
                        overlap >= self.shared_pages_threshold
                        and containment >= self.containment_threshold
                    ):
                        yield left_idx, right_idx, overlap

    @staticmethod
    def _count_union_components(booklet_union: "UnionFind", booklet_count: int) -> int:
        roots = {booklet_union.find(booklet_idx) for booklet_idx in range(booklet_count)}
        return len(roots)

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        logger = deps()["logger"]
        if data is not None:
            yield from data
        assert world_size == 1, "RewriteRemoveIdsByBooklet must run with a single task."
        self.output_folder.mkdir(parents=True, exist_ok=True)

        cluster_members, needed_doc_idxs, skipped_cluster_paths = self._collect_cluster_members()
        logger.info(
            "Collected cluster members: cluster_count={} referenced_doc_refs={} skipped_cluster_paths={}",
            len(cluster_members),
            sum(len(doc_idxs) for doc_idxs in needed_doc_idxs.values()),
            skipped_cluster_paths,
        )
        doc_to_booklet, booklet_docs, booklet_meta = self._load_documents(
            needed_doc_idxs=needed_doc_idxs
        )
        booklet_ids = list(booklet_docs)
        booklet_idx_by_id = {booklet_id: idx for idx, booklet_id in enumerate(booklet_ids)}
        booklet_sizes = [len(booklet_docs[booklet_id]) for booklet_id in booklet_ids]
        logger.info(
            "Loaded booklet mapping: input_files={} total_doc_refs={} total_booklets={} source_remove_folder={} cluster_count={}",
            len(self.source_files),
            len(doc_to_booklet),
            len(booklet_docs),
            self.source_remove_folder,
            len(cluster_members),
        )

        booklet_union = UnionFind()
        for booklet_idx in range(len(booklet_ids)):
            booklet_union.find(booklet_idx)
        merged_pair_count = 0
        for left_idx, right_idx, _overlap in self._iter_shared_page_overlaps(
            cluster_members=cluster_members,
            doc_to_booklet=doc_to_booklet,
            booklet_idx_by_id=booklet_idx_by_id,
            booklet_sizes=booklet_sizes,
            logger=logger,
        ):
            booklet_union.union(left_idx, right_idx)
            merged_pair_count += 1
        logger.info(
            "Connected {} booklet pairs. component_count={}",
            merged_pair_count,
            self._count_union_components(booklet_union, len(booklet_ids)),
        )

        booklet_components: dict[int, list[int]] = {}
        for booklet_idx in range(len(booklet_ids)):
            booklet_components.setdefault(booklet_union.find(booklet_idx), []).append(booklet_idx)
        logger.info("Built {} booklet components.", len(booklet_components))

        dropped_booklets: set[str] = set()
        winners: dict[int, str] = {}
        for component_root, component_booklet_idxs in booklet_components.items():
            winner = max(
                component_booklet_idxs,
                key=lambda booklet_idx: self._winner_key(
                    booklet_idx, booklet_ids, booklet_docs, booklet_meta
                ),
            )
            winner_id = booklet_ids[winner]
            winners[component_root] = winner_id
            for booklet_idx in component_booklet_idxs:
                booklet_id = booklet_ids[booklet_idx]
                if booklet_id != winner_id:
                    dropped_booklets.add(booklet_id)

        remove_by_rank: dict[int, set[int]] = {
            task_rank: set() for task_rank in range(self.total_tasks)
        }
        for booklet_id in dropped_booklets:
            for file_rank, doc_idx in booklet_docs[booklet_id]:
                remove_by_rank[file_rank].add(doc_idx)

        booklet_score_cache = {
            booklet_id: self._winner_key(booklet_idx, booklet_ids, booklet_docs, booklet_meta)
            for booklet_idx, booklet_id in enumerate(booklet_ids)
        }
        skipped_surviving_refs = 0
        inconsistent_surviving_refs = 0
        for cluster_id, members in cluster_members.items():
            surviving_docs: list[tuple[tuple[int, int], str]] = []
            for doc_ref in members:
                booklet_id = doc_to_booklet.get(doc_ref)
                if booklet_id is None:
                    skipped_surviving_refs += 1
                    continue
                if booklet_id in dropped_booklets:
                    continue
                surviving_docs.append((doc_ref, booklet_id))
            if not surviving_docs:
                continue
            missing_booklet_ids = [
                doc_ref
                for doc_ref, booklet_id in surviving_docs
                if booklet_id not in booklet_score_cache
            ]
            if missing_booklet_ids:
                inconsistent_surviving_refs += len(missing_booklet_ids)
                logger.warning(
                    "Skipping {} surviving docs in cluster_id={} due to missing booklet scores: {}",
                    len(missing_booklet_ids),
                    cluster_id,
                    missing_booklet_ids[:10],
                )
                surviving_docs = [
                    (doc_ref, booklet_id)
                    for doc_ref, booklet_id in surviving_docs
                    if booklet_id in booklet_score_cache
                ]
                if not surviving_docs:
                    continue
            representative = max(
                surviving_docs,
                key=lambda item: (
                    booklet_score_cache[item[1]],
                    -item[0][0],
                    -item[0][1],
                ),
            )
            representative_doc_ref = representative[0]
            for (file_rank, doc_idx), _booklet_id in surviving_docs:
                if (file_rank, doc_idx) != representative_doc_ref:
                    remove_by_rank[file_rank].add(doc_idx)

        for task_rank in range(self.total_tasks):
            src_clusters = self.source_remove_folder / f"{task_rank:06d}.clusters"
            src_sizes = self.source_remove_folder / f"{task_rank:06d}.sizes"
            dst_clusters = self.output_folder / f"{task_rank:06d}.clusters"
            dst_sizes = self.output_folder / f"{task_rank:06d}.sizes"
            if src_clusters.exists():
                shutil.copyfile(src_clusters, dst_clusters)
            else:
                dst_clusters.write_bytes(b"")
            if src_sizes.exists():
                shutil.copyfile(src_sizes, dst_sizes)
            else:
                dst_sizes.write_bytes(b"")
            _write_binary_uints(
                self.output_folder / f"{task_rank:06d}.remove",
                sorted(remove_by_rank.get(task_rank, set())),
            )

        logger.info(
            "Booklet rewrite completed: total_booklets={} dropped_booklets={} merged_pairs={} skipped_cluster_paths={} skipped_surviving_refs={}",
            len(booklet_docs),
            len(dropped_booklets),
            merged_pair_count,
            skipped_cluster_paths,
            skipped_surviving_refs,
        )
        if inconsistent_surviving_refs:
            logger.warning(
                "Encountered {} surviving docs with inconsistent booklet score mappings.",
                inconsistent_surviving_refs,
            )
        logger.info("Booklet winners selected for {} components.", len(winners))
        return


def resolve_execution_mode(args, total_tasks: int, LocalPipelineExecutor, SlurmPipelineExecutor):
    logger = deps()["logger"]
    sbatch_path = shutil.which("sbatch")
    if args.no_slurm:
        use_slurm = False
    elif args.use_slurm:
        use_slurm = True
    else:
        use_slurm = False

    if use_slurm and sbatch_path is None:
        raise RuntimeError("--use-slurm requires sbatch")

    default_workers = max(1, os.cpu_count() or 1)
    workers = default_workers if args.workers <= 0 else args.workers
    use_parallel_local = not args.force_serial and workers > 1

    if use_slurm:
        logger.info("Using SlurmPipelineExecutor with cpus_per_task={}.", max(1, workers))
        return use_slurm, workers, SlurmPipelineExecutor, {}

    if use_parallel_local:
        logger.info(
            "Using LocalPipelineExecutor with {} worker(s) (start_method={}).",
            min(workers, total_tasks),
            args.start_method,
        )
        return False, workers, LocalPipelineExecutor, {"start_method": args.start_method}

    logger.info("Using SerialLocalPipelineExecutor (workers forced to 1).")
    return False, 1, SerialLocalPipelineExecutor, {}


def make_executor(
    *,
    use_slurm: bool,
    executor_cls,
    executor_extra_kwargs: dict,
    workers: int,
    args,
    log_root: Path,
    slurm_log_dir: Path,
    pipeline,
    tasks: int,
    logging_subdir: str,
    job_name: str,
    depends=None,
):
    log_dir = log_root / logging_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    if use_slurm:
        return executor_cls(
            pipeline=pipeline,
            tasks=tasks,
            logging_dir=str(log_dir),
            depends=depends,
            job_name=job_name,
            partition=args.partition,
            time=args.time,
            slurm_logs_folder=str(slurm_log_dir),
            venv_path=args.venv,
            cpus_per_task=max(1, workers),
        )

    return executor_cls(
        pipeline=pipeline,
        tasks=tasks,
        workers=min(max(1, workers), tasks) if tasks > 0 else 1,
        logging_dir=str(log_dir),
        depends=depends,
        **executor_extra_kwargs,
    )


def build_pipeline(args):
    dep = deps()
    LocalPipelineExecutor = dep["LocalPipelineExecutor"]
    SlurmPipelineExecutor = dep["SlurmPipelineExecutor"]
    PipelineStepBase = dep["PipelineStep"]
    MinhashDedupSignature = dep["MinhashDedupSignature"]
    MinhashConfig = dep["MinhashConfig"]
    MinhashDedupBuckets = dep["MinhashDedupBuckets"]
    MinhashDedupCluster = dep["MinhashDedupCluster"]
    MinhashDedupFilter = dep["MinhashDedupFilter"]
    JsonlReader = dep["JsonlReader"]
    JsonlWriter = dep["JsonlWriter"]
    HashConfig = dep["HashConfig"]
    Languages = dep["Languages"]

    global PageAuditStep, PageDecisionWriter
    if PageAuditStep is object:

        class _PageAuditStep(PipelineStepBase):
            type = "📝 - AUDIT"
            name = "Page decision logger"

            def __init__(self, audit_logger: PageAuditLogger):
                super().__init__()
                self.audit_logger = audit_logger

            def run(self, data, rank: int = 0, world_size: int = 1):
                for shard_id, doc in enumerate(data):
                    with self.track_time():
                        self.audit_logger.log(doc, rank, decision="kept", shard_id=shard_id)
                        yield doc

        class _PageDecisionWriter:
            def __init__(self, audit_logger: PageAuditLogger | None = None):
                self.audit_logger = audit_logger

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, document, rank: int = 0, **kwargs):
                if self.audit_logger is not None:
                    self.audit_logger.log(document, rank, decision="removed")

        PageAuditStep = _PageAuditStep
        PageDecisionWriter = _PageDecisionWriter

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    all_files = collect_input_files(input_path)
    total_tasks = len(all_files)
    if total_tasks == 0:
        raise ValueError(f"No files found under input path: {args.input}")
    input_file_list = collect_stage3_source_files(input_path, all_files)

    minhash_dirname = (
        f"minhash-{args.ngram}gram-{args.buckets}buckets-{args.hashes_per_bucket}hashes"
    )
    minhash_dir = Path(args.output) / minhash_dirname
    result_dir = minhash_dir / "results"
    log_dir = minhash_dir / "logs"
    slurm_log_dir = Path(args.slurm_logs) if args.slurm_logs else minhash_dir / "slurm_logs"
    audit_dir = result_dir / "page_audit"

    if input_path.is_relative_to(minhash_dir.resolve()) or (
        input_path.is_dir() and minhash_dir.resolve().is_relative_to(input_path)
    ):
        raise ValueError("Input and MinHash work directories must not contain each other")

    if args.force_clean and minhash_dir.exists():
        shutil.rmtree(minhash_dir, ignore_errors=True)

    minhash_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_pages:
        audit_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 and the booklet rewrite must use exactly the same ordered shard set.
    paths_file = None
    if input_path.is_dir():
        paths_file = minhash_dir / "input-files.txt"
        paths_file.write_text("\n".join(input_file_list) + "\n", encoding="utf-8")
    input_reader = JsonlReader(
        str(input_path),
        paths_file=str(paths_file) if paths_file else None,
        doc_progress=not args.audit_pages,
    )
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        n_grams=args.ngram,
        num_buckets=args.buckets,
        hashes_per_bucket=args.hashes_per_bucket,
    )

    use_slurm, workers, executor_cls, executor_extra_kwargs = resolve_execution_mode(
        args,
        total_tasks,
        LocalPipelineExecutor,
        SlurmPipelineExecutor,
    )
    audit_logger = PageAuditLogger(str(audit_dir)) if args.audit_pages else None

    def executor(**kwargs):
        return make_executor(
            use_slurm=use_slurm,
            executor_cls=executor_cls,
            executor_extra_kwargs=executor_extra_kwargs,
            workers=workers,
            args=args,
            log_root=log_dir,
            slurm_log_dir=slurm_log_dir,
            **kwargs,
        )

    stage1 = executor(
        pipeline=[
            input_reader,
            MinhashDedupSignature(
                output_folder=str(result_dir / "signatures"),
                config=minhash_config,
                language=Languages.japanese,
                skip_existing_sigs=True,
            ),
        ],
        tasks=total_tasks,
        logging_subdir="signatures",
        job_name="minhash_signatures",
    )

    stage2 = executor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(result_dir / "signatures"),
                output_folder=str(result_dir / "buckets"),
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        depends=stage1,
        logging_subdir="buckets",
        job_name="minhash_buckets",
    )

    stage3 = executor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(result_dir / "buckets"),
                output_folder=str(result_dir / "remove_ids"),
                config=minhash_config,
                save_cluster_id=True,
                save_cluster_size=True,
            ),
        ],
        tasks=1,
        depends=stage2,
        logging_subdir="clusters",
        job_name="minhash_clusters",
    )

    stage3_5 = executor(
        pipeline=[
            RewriteRemoveIdsByBooklet(
                source_input_path=args.input,
                source_files=input_file_list,
                source_remove_folder=str(result_dir / "remove_ids"),
                output_folder=str(result_dir / "remove_ids_booklet"),
                total_tasks=total_tasks,
                shared_pages_threshold=args.booklet_shared_pages_threshold,
                containment_threshold=args.booklet_containment_threshold,
            ),
        ],
        tasks=1,
        depends=stage3,
        logging_subdir="rewrite_remove_ids",
        job_name="minhash_rewrite_remove_ids",
    )

    filter_pipeline = [
        input_reader,
        MinhashDedupFilter(
            input_folder=str(result_dir / "remove_ids_booklet"),
            exclusion_writer=PageDecisionWriter(
                audit_logger=audit_logger,
            )
            if audit_logger is not None
            else None,
            load_cluster_ids=True,
            load_cluster_sizes=True,
        ),
    ]
    if audit_logger is not None:
        filter_pipeline.append(PageAuditStep(audit_logger))
    if args.write_deduplicated_output:
        filter_pipeline.append(JsonlWriter(output_folder=str(result_dir / "deduplicated_output")))

    stage4 = executor(
        pipeline=filter_pipeline,
        tasks=total_tasks,
        depends=stage3_5,
        logging_subdir="filter",
        job_name="minhash_filter",
    )

    return stage4, audit_logger
