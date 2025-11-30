import argparse
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_tokens(token_path: Path) -> list[str]:
    tokens = np.load(token_path, allow_pickle=True).tolist()
    if not isinstance(tokens, list):
        raise ValueError("decoded_tokens.npy did not contain a list")
    return tokens


def find_token_index(tokens: list[str], target: str) -> int:
    """Find the first token matching target (case-insensitive, strips BPE markers)."""
    target = target.lower()

    def normalize(tok: str) -> str:
        return tok.replace("Ġ", "").replace("Ċ", "").replace("▁", "").lower()

    for i, tok in enumerate(tokens):
        if normalize(tok) == target:
            return i
    raise ValueError(f"Token '{target}' not found in decoded_tokens.npy")


def build_patch_map(tokens: list[str]) -> tuple[list[tuple[int, int, int]], int, int, int]:
    """
    Map <row_i_col_j> tokens to the start index of their 8x8 sub-patches.
    Returns: (mapping, num_rows, num_cols, sub_side)
    """
    pattern = re.compile(r"<row_(\d+)_col_(\d+)>")
    mapping: list[tuple[int, int, int]] = []
    max_row = 0
    max_col = 0

    sub_count = 64  # expect 8x8 subpatches following each patch token
    sub_side = int(math.isqrt(sub_count))

    for idx, tok in enumerate(tokens):
        m = pattern.match(tok)
        if m:
            row = int(m.group(1))
            col = int(m.group(2))
            start = idx + 1
            end = start + sub_count
            if end > len(tokens):
                continue
            mapping.append((row, col, start))
            max_row = max(max_row, row)
            max_col = max(max_col, col)

    if not mapping:
        raise ValueError("No <row_?_col_?> patch tokens found in decoded_tokens.npy")

    return mapping, max_row, max_col, sub_side


def entropy(prob: np.ndarray) -> float:
    prob = prob + 1e-12
    return float(-(prob * np.log(prob)).sum())


def load_attention_layer(attn_dir: Path, layer: int) -> np.ndarray:
    path = attn_dir / f"attention_fp16_rounded_layer_{layer}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing attention file: {path}")
    data = np.load(path)
    attn = data["attention"]
    # expected shape: [1, heads, query, key]
    return attn


def main():
    parser = argparse.ArgumentParser(
        description="Build a token attention heatmap using attention sum and spatial entropy weighting."
    )
    parser.add_argument(
        "--token",
        default="jacket",
        help="Token text to search for (case-insensitive, strips Ġ/Ċ/▁ markers).",
    )
    parser.add_argument(
        "--decoded-tokens",
        type=Path,
        default=Path("decoded_tokens.npy"),
        help="Path to decoded_tokens.npy",
    )
    parser.add_argument(
        "--attn-dir",
        type=Path,
        default=Path("attentions_5"),
        help="Directory containing attention_fp16_rounded_layer_*.npz",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0-29",
        help="Layers to include, e.g., '0-29' or '0,5,10'.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Keep only top-K attention targets per head before aggregation (0 to disable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("token_heatmap.png"),
        help="Output heatmap image path.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="Optional output path for overlay on image.png",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Alpha for heatmap overlay (0-1).",
    )
    args = parser.parse_args()

    tokens = load_tokens(args.decoded_tokens)
    token_idx = find_token_index(tokens, args.token)
    patch_map, num_rows, num_cols, sub_side = build_patch_map(tokens)

    # Parse layers
    if "-" in args.layers:
        start, end = args.layers.split("-")
        layer_list = list(range(int(start), int(end) + 1))
    else:
        layer_list = [int(x) for x in args.layers.split(",") if x]

    agg_heatmap = np.zeros((num_rows * sub_side, num_cols * sub_side), dtype=np.float64)
    total_weight = 0.0

    for layer in layer_list:
        attn = load_attention_layer(args.attn_dir, layer)
        if attn.ndim != 4:
            raise ValueError(f"Unexpected attention shape in layer {layer}: {attn.shape}")
        heads = attn.shape[1]

        for head in range(heads):
            vec = attn[0, head, token_idx, :].astype(np.float64)
            attn_sum = float(vec.sum())
            if attn_sum <= 0:
                continue
            prob = vec / attn_sum

            # Top-K filtering (set others to zero)
            if args.topk > 0 and args.topk < prob.size:
                k = args.topk
                # argsort descending
                top_idx = np.argpartition(prob, -k)[-k:]
                mask = np.zeros_like(prob)
                mask[top_idx] = 1.0
                prob = prob * mask
                # re-normalize to sum=1 over kept entries
                prob_sum = prob.sum()
                if prob_sum > 0:
                    prob /= prob_sum
                else:
                    continue

            # Spatial entropy weight (lower entropy -> higher weight)
            ent = entropy(prob)
            max_ent = math.log(prob.size)
            ent_weight = max(0.0, 1.0 - ent / max_ent)
            weight = attn_sum * ent_weight
            if weight <= 0:
                continue

            grid = np.zeros_like(agg_heatmap)
            for row, col, start in patch_map:
                indices = slice(start, start + sub_side * sub_side)
                patch_probs = prob[indices].reshape(sub_side, sub_side)
                r0 = (row - 1) * sub_side
                c0 = (col - 1) * sub_side
                grid[r0 : r0 + sub_side, c0 : c0 + sub_side] = patch_probs

            agg_heatmap += weight * grid
            total_weight += weight

    if total_weight == 0:
        raise RuntimeError("No attention weight accumulated; check token and inputs.")

    agg_heatmap /= total_weight

    plt.figure(figsize=(num_cols, num_rows))
    plt.imshow(agg_heatmap, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Weighted attention")
    plt.title(f"Token '{args.token}' aggregated heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved heatmap to {args.output}")

    # Optional overlay on image.png
    if args.overlay:
        from PIL import Image
        base_path = Path("image.png")
        if not base_path.exists():
            raise FileNotFoundError("image.png not found for overlay")
        base = Image.open(base_path).convert("RGBA")

        # Resize heatmap to base size
        heatmap_norm = (agg_heatmap - agg_heatmap.min()) / (np.ptp(agg_heatmap) + 1e-12)
        heatmap_img = plt.get_cmap("hot")(heatmap_norm)  # RGBA float in [0,1]
        heatmap_img[..., 3] = args.overlay_alpha
        heatmap_img = (heatmap_img * 255).astype(np.uint8)
        # Keep block structure: nearest-neighbor upsampling
        heatmap_pil = Image.fromarray(heatmap_img, mode="RGBA").resize(base.size, resample=Image.NEAREST)

        overlay_img = Image.alpha_composite(base, heatmap_pil)
        overlay_img.save(args.overlay)
        print(f"Saved overlay to {args.overlay}")


if __name__ == "__main__":
    main()
