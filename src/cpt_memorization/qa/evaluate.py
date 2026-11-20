"""Judge generated answers against their source context and gold answer."""

from .chat import run


def main():
    return run("evaluate")


if __name__ == "__main__":
    main()
