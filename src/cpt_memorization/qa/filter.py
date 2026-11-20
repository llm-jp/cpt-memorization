"""Select document snippets suitable for grounded QA."""

from .chat import run


def main():
    return run("filter")


if __name__ == "__main__":
    main()
