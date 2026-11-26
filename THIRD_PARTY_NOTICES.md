# Source attribution

The publication-level and page-level MinHash implementations in
`src/cpt_memorization/corpus/dedup/minhash_dedup.py` and
`src/cpt_memorization/corpus/dedup/_minhash_pages.py` adapt the
[llm-jp/scripts MinHash pipeline](https://github.com/llm-jp/scripts/blob/main/corpus/llm-jp-corpus-v4/common/dedup/minhash/slurm/minhash_dedup.py).

These two modified source files retain the upstream Apache License 2.0.
The full license is included in [LICENSE-APACHE](LICENSE-APACHE).

The deduplication pipeline uses DataTrove. The inference extra uses vLLM,
Transformers, and OpenAI Harmony. These dependencies and their weights are
distributed separately. For the project's licensing terms, see
[LICENSE_STATUS.md](LICENSE_STATUS.md).
