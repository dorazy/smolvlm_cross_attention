import argparse
import numpy as np


def clean_token(tok: str) -> str:
    """Replace common BPE markers for display only."""
    return tok.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")


def main():
    parser = argparse.ArgumentParser(description="Inspect decoded_tokens.npy")
    parser.add_argument("path", nargs="?", default="decoded_tokens.npy", help="Path to decoded_tokens.npy")
    parser.add_argument("--limit", type=int, default=50, help="How many tokens to show (after cleaning)")
    parser.add_argument("--raw", action="store_true", help="Show raw tokens without cleaning")
    args = parser.parse_args()

    toks = np.load(args.path, allow_pickle=True)
    print(f"file: {args.path}")
    print(f"shape: {toks.shape}, dtype: {toks.dtype}")
    print(f"total tokens: {len(toks)}")

    to_show = toks[: args.limit]
    if args.raw:
        display = to_show
    else:
        display = [clean_token(t) for t in to_show]

    print("\n=== sample tokens ===")
    for i, t in enumerate(display, 1):
        print(f"{i:3d}: {repr(t)}")


if __name__ == "__main__":
    main()
