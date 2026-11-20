# Relationship to the paper

**Analyzing Task Performance and Memorization Risk in Continued Pretraining of LLMs**
was accepted to the EACL 2027 Main Conference. The study uses Japanese government
publications provided by the National Diet Library of Japan to examine how
continued pretraining affects question answering and reproduction of training
passages. The table maps the GovPub experiments to their commands and inputs.

| Paper component | Implementation / artifact | Verification and external inputs |
| --- | --- | --- |
| §3.2 GovPub preprocessing | `corpus-extract`, `corpus-dedup`, `corpus-language` | Synthetic OCR/dedup/language integration check; original OCR and lid weights required for full corpus |
| §3.3–3.4 CPT | External Megatron-LM | Requires external training, mixing and checkpoint-conversion tools |
| §4.1 QA construction, Appendix A | `qa-filter`, `qa-synthesize`, bundled prompt YAMLs | CPU orchestration tests; GPT-OSS and source corpus required for inference |
| §4.1 open-/closed-book answers and judging | `qa-answer`, `qa-evaluate` | Three few-shot demonstrations by default; explicit compatible tokenizer and seed settings |
| §4.2 / Figure 3 temperature sweep | `qa-sweep`, `qa-aggregate`, `qa-plot` | Recomputed all 96 supplied corrected runs; context-free tables in [results](results) |
| §4.3 / Figure 4 general evaluation | External llm-jp-eval v1.4.1 | Requires the runner, benchmark configs and original scores |
| §5.1 sampling and greedy continuation | `memory-sample`, `memory-generate` | Shared anchors, token IDs, and tokenizer; N=1000, M=50 for main figures |
| §5.2 / Figure 6 and Appendix B | `memory-paired`, `memory-sweep`, `memory-plot` | Full paired inputs and BASE exclusion; synthetic numerical regression checks |
| §5.3 examples / manual inspection | `memory-split`, `memory-extract`, `memory-examples` | Original completion files and manual annotation required |
| §5.4 prefix copying and residual set | `memory-containment`, `memory-residual` | Token 3-gram multiset containment at 0.5; residual/control query export |
| §5.4 near-duplicate frequencies | `neardup-summary` and external search | Source corpus/index and historical count outputs absent; no claim to recover reported medians |

The paper reports 300,821 source publications, 253,639 retained publications,
approximately 13.7B target tokens, and 9,890 sampled memorization examples.
Corpus and sample sizes depend on the inputs.
The main memorization comparison excludes BASE-memorized examples. The original
sampled anchors and completions are external experiment inputs.
Regenerating an equal-size random sample does not recover their historical identity.

The supplied corrected QA sweep contains 2,340 candidate QA IDs and 224,622
successful judgments from 96 model-runs. Their global intersection is 2,328 IDs,
matching the final comparison size. Recomputing with the reorganized aggregator
produced 36 aggregate rows; mean and standard deviation agree with the supplied
rounded table within 4.84e-11 on the 0–1 scale. The raw source contexts, generated
answers, and excluded item identifiers are not distributed under `docs/results`.

For the released accuracy tables there was one run at zero temperature and
three at each nonzero temperature. The corresponding configuration is
`qa_sweep.json`. The alternative `qa_sweep_paper.json` requests three runs at
every temperature (108 runs). For a single run, the reported standard deviation
of zero is a convention. Run-to-run variation at that temperature cannot be
estimated from the table.

The historical QA records used two OCR source layouts. The packaged extractor
reads NDLOCR tar members. To reconstruct the original dataset, prepare other
layouts in the same document schema and supply the original document order,
sampled segments and model runtime. GPU topology and inference-library versions
can affect regeneration even with a fixed decoding seed.

## Interpretation of the outputs

- **GovPub preprocessing.** The paper deduplicates publication records and
  retains Japanese publications. The QA construction commands consume these
  retained shards. Page-level MinHash, LZ77 and page cleaning are optional
  analyses.
- **Memorization scores.** The code uses unsmoothed effective-order token-ID
  BLEU and clipped multiset token 3-gram containment. Repeated n-grams can
  affect containment. Confirming the original 45 of 56 prefix-overlap cases and
  the 11 residual cases requires the original completions and scoring settings.
- **Near-duplicate counts.** The exact scanner counts token windows in the
  provided corpus. Verifying the reported near-duplicate medians of 30 and
  1,613 requires the original approximate-search criterion, index and counts.
- **Qualitative analysis.** Candidate extraction identifies cases for inspection.
  The review of 300 examples and the interpretation of boundary cases require the
  original human annotations.
- **Training and general-domain evaluation.** Reproducing the training curves
  and general-domain comparison requires the first-phase BASE checkpoint,
  the GovOnly, GovMix and CONTROL training configurations, llm-jp-eval v1.4.1
  with its category macro averaging, and the per-checkpoint scores.

CPU checks cover 1% document sampling, random segments of up to 1,000 characters,
selection and synthesis prompts, distinct few-shot demonstrations and
open-/closed-book answers. The memorization checks cover the five snippet lengths
(200, 400, 600, 800 and 1,000), the 50-token suffix, paired BASE exclusion and
residual/control selection. These checks validate the mechanics on synthetic data.
The original sampled IDs and generated token sequences require the historical
corpus, tokenizer, checkpoints and inference environment.

See [the workflow guide](REPRODUCING.md) for executable commands and required
inputs, and [the validation record](VALIDATION.md) for the tests performed.
