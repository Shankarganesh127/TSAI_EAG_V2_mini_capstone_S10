from __future__ import annotations

import argparse

from .core import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic LLM caller for local and hosted providers"
    )
    parser.add_argument("prompt", help="Prompt text to send to the model")
    args = parser.parse_args()

    client = LLMClient()
    print(client.chat(args.prompt))


if __name__ == "__main__":
    main()
