"""Extract BLEU >= .75 and best-window token 10-gram Jaccard < .6 rows."""

from .extract_memorized_samples import main as extract


def main():
    return extract(default_jaccard_threshold=0.6, default_variant_source="parent-dir")


if __name__ == "__main__":
    main()
