import math
from collections import Counter
from typing import AbstractSet, Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar


Token = TypeVar("Token", bound=Hashable)


def token_bleu_score(reference: Sequence[Token], candidate: Sequence[Token]) -> float:
    """Return token BLEU on a 0-1 scale without smoothing."""
    if not candidate:
        return 0.0

    max_n = min(4, len(candidate), len(reference))
    if max_n == 0:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        cand_ngrams = Counter(tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1))
        total = sum(cand_ngrams.values())
        if total == 0:
            return 0.0
        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
        precisions.append(overlap / total)

    if any(p == 0.0 for p in precisions):
        return 0.0

    brevity_penalty = 1.0
    if len(candidate) < len(reference):
        brevity_penalty = math.exp(1.0 - float(len(reference)) / len(candidate))

    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def set_jaccard(
    set_a: AbstractSet[Hashable],
    set_b: AbstractSet[Hashable],
    empty_score: float = 0.0,
) -> float:
    """Return unweighted Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return empty_score
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def weighted_jaccard(counter_a: Counter, counter_b: Counter) -> float:
    """Return weighted Jaccard similarity between two multisets."""
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 0.0
    numerator = sum(min(counter_a[k], counter_b[k]) for k in keys)
    denominator = sum(max(counter_a[k], counter_b[k]) for k in keys)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def token_ngrams(tokens: Sequence[Token], n: int) -> List[Tuple[Token, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_jaccard(tokens_a: Sequence[Token], tokens_b: Sequence[Token], n: int) -> float:
    """Return weighted Jaccard similarity between token n-gram multisets."""
    return weighted_jaccard(Counter(token_ngrams(tokens_a, n)), Counter(token_ngrams(tokens_b, n)))


def shared_multiset_count(counter_a: Counter, counter_b: Counter) -> int:
    """Return the multiset intersection size between two counters."""
    return sum(min(counter_a[k], counter_b[k]) for k in set(counter_a) | set(counter_b))


def find_best_jaccard_window(
    prefix_tokens: Sequence[Token],
    suffix_tokens: Sequence[Token],
    ngram_size: int,
) -> Dict[str, float]:
    suffix_len = len(suffix_tokens)
    empty = {
        "best_window_start": -1,
        "best_window_end": -1,
        "best_window_length": 0,
        "best_window_bleu": 0.0,
        "best_shared_ngram_count": 0,
        "jaccard_score": 0.0,
    }

    if not prefix_tokens or not suffix_tokens or len(prefix_tokens) < suffix_len:
        return empty

    target_counter = Counter(token_ngrams(suffix_tokens, ngram_size))
    if not target_counter:
        return empty

    best = empty.copy()
    for start in range(len(prefix_tokens) - suffix_len + 1):
        end = start + suffix_len
        window_tokens = prefix_tokens[start:end]
        window_counter = Counter(token_ngrams(window_tokens, ngram_size))
        score = weighted_jaccard(target_counter, window_counter)
        bleu = token_bleu_score(suffix_tokens, window_tokens)
        shared_count = shared_multiset_count(target_counter, window_counter)

        if score > best["jaccard_score"] or (
            score == best["jaccard_score"] and bleu > best["best_window_bleu"]
        ):
            best = {
                "best_window_start": start,
                "best_window_end": end,
                "best_window_length": suffix_len,
                "best_window_bleu": bleu,
                "best_shared_ngram_count": shared_count,
                "jaccard_score": score,
            }

    return best


def compute_memorization_metrics(
    suffix_tokens: Sequence[Token],
    predicted_tokens: Sequence[Token],
    prefix_tokens: Optional[Sequence[Token]] = None,
    ngram_size: int = 10,
) -> Dict[str, float]:
    """Compute the canonical BLEU and prefix/suffix Jaccard metrics for a sample."""
    metrics = find_best_jaccard_window(prefix_tokens or [], suffix_tokens, ngram_size)
    return {
        "bleu": token_bleu_score(suffix_tokens, predicted_tokens),
        **metrics,
    }
