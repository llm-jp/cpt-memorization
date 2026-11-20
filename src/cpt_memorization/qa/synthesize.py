"""Synthesize one QA pair per selected snippet."""

from .chat import run


def main():
    return run("synthesize")


if __name__ == "__main__":
    main()
