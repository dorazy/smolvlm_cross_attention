import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect attention npz file.")
    parser.add_argument("path", help="Path to attentions_5/attention_fp16_rounded_layer_X.npz")
    parser.add_argument("--head", type=int, default=0, help="Head index to sample print")
    parser.add_argument("--rows", type=int, default=3, help="Rows to print from sample slice")
    args = parser.parse_args()

    data = np.load(args.path)
    if "attention" not in data:
        raise ValueError(f"'attention' key not found in {args.path}")

    attn = data["attention"]
    print(f"file: {args.path}")
    print(f"keys: {list(data.keys())}")
    print(f"shape: {attn.shape}, dtype: {attn.dtype}")
    print(f"min: {attn.min()}, max: {attn.max()}")

    h = min(args.head, attn.shape[1] - 1)
    sample = attn[0, h, : args.rows, : args.rows]
    print(f"\nhead {h} top-left {args.rows}x{args.rows} slice:\n{sample}")


if __name__ == "__main__":
    main()
