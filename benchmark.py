import argparse
import base64
import gc
import json
import math
import os
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Reuse most of the logic from app.py but run as an offline benchmark task.

decoded_tokens = None
smolvlm_patch_size = 512
smolvlm_token_patch_size = 8


def load_data():
    """Load decoded tokens once."""
    global decoded_tokens
    try:
        decoded_tokens = np.load("decoded_tokens.npy").tolist()
        return True
    except FileNotFoundError as e:
        print(f"Error loading tokens: {e}")
        return False


def load_attention_layer(layer_index):
    """Load attention data for a specific layer."""
    filename = f"attentions_5/attention_fp16_rounded_layer_{layer_index}.npz"
    if not os.path.exists(filename):
        return None
    try:
        return np.load(filename)["attention"]
    except Exception as e:
        print(f"Failed to load attention layer {layer_index}: {e}")
        return None


def build_image_patch_dict(tokens):
    """Map patch tokens to indices."""
    patch_indices = {}
    for i, word in enumerate(tokens):
        if "row" in word or "global" in word:
            patch_indices[word] = [i + 1 + j for j in range(64)]
    return patch_indices


def reshape_patch(patch):
    """Expand token patch to image patch size."""
    expanded_size = smolvlm_token_patch_size ** 2
    expanded_patch = np.zeros((smolvlm_patch_size, smolvlm_patch_size), dtype=np.float32)
    square_patch = patch.reshape(smolvlm_token_patch_size, smolvlm_token_patch_size)

    for i in range(smolvlm_token_patch_size):
        for j in range(smolvlm_token_patch_size):
            value = square_patch[i, j]
            expanded_patch[
                i * expanded_size : i * expanded_size + expanded_size,
                j * expanded_size : j * expanded_size + expanded_size,
            ] = value

    del square_patch
    return expanded_patch


def get_attn_map(word_attn, head, patch_indices, num_rows=3, num_cols=4):
    """Generate attention map for a given head."""
    attn = word_attn[head, :]
    final_heatmap = np.zeros((512 * num_rows, 512 * num_cols), dtype=np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            patch_key = f"<row_{i+1}_col_{j+1}>"
            if patch_key in patch_indices:
                indices = patch_indices[patch_key]
                attn_patch = attn[indices]
                p = reshape_patch(attn_patch)
                final_heatmap[i * 512 : i * 512 + 512, j * 512 : j * 512 + 512] = p
                del attn_patch, p

    del attn
    return final_heatmap


def create_attention_heatmap(final_heatmap, image, threshold=0.0):
    """Create heatmap PNG (base64) and return dimensions."""
    image_w, image_h = image.size
    resized_w = math.ceil(image_w / smolvlm_patch_size) * smolvlm_patch_size
    resized_h = math.ceil(image_h / smolvlm_patch_size) * smolvlm_patch_size

    original_min = float(final_heatmap.min())
    original_max = float(final_heatmap.max())

    if threshold > 0.0:
        final_heatmap[final_heatmap < threshold] = 0.0

    heatmap_min = final_heatmap.min()
    heatmap_max = final_heatmap.max()
    heatmap_range = heatmap_max - heatmap_min

    if heatmap_range == 0:
        final_heatmap.fill(0.5)
        normalized_heatmap = final_heatmap
    else:
        final_heatmap -= heatmap_min
        final_heatmap /= heatmap_range
        normalized_heatmap = final_heatmap

    fig = plt.figure(figsize=(resized_w / 100, resized_h / 100), dpi=100)
    plt.imshow(normalized_heatmap, cmap="hot", alpha=0.7, interpolation="nearest")
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, transparent=True, dpi=100)
    buffer.seek(0)
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close(fig)
    plt.clf()

    del normalized_heatmap, final_heatmap
    gc.collect()

    stats = {"min": original_min, "max": original_max}
    return heatmap_base64, stats, resized_w, resized_h


def normalize_token(token: str) -> str:
    return token.strip().lower().strip(".,?!")


def find_token_index(main_word: str, tokens):
    target = normalize_token(main_word)
    for idx, tok in enumerate(tokens):
        if "row" in tok or "global" in tok or tok.startswith("<"):
            continue
        if normalize_token(tok) == target:
            return idx
    return None


def get_word_attention(word, tokens, layer_index, word_index=None):
    """Return attention slice and index for a word on a specific layer."""
    attention_layer = load_attention_layer(layer_index)
    if attention_layer is None:
        return None, None

    try:
        if word_index is None:
            word_index = tokens.index(word)

        word_attn = attention_layer[0, :, word_index, :].copy()
        del attention_layer
        gc.collect()
        return word_attn, word_index
    except Exception:
        del attention_layer
        return None, None


def extract_target_word(question_text: str) -> str | None:
    """Extract the main object after 'Is there a/an ...'."""
    text = question_text.lower().replace("?", "").replace(",", "")
    markers = ["is there a ", "is there an "]
    for marker in markers:
        if marker in text:
            tail = text.split(marker, 1)[1]
            parts = tail.split()
            if parts:
                return parts[0]
    return None


def tokens_to_text(tokens):
    """Create a readable assistant response from tokens."""
    content_tokens = []
    for tok in tokens:
        if "row" in tok or "global" in tok or tok.startswith("<"):
            continue
        if tok == "\n":
            content_tokens.append("\n")
        elif tok.strip() != "":
            content_tokens.append(tok)
    return "".join(content_tokens)


def overlay_heatmap_on_image(base_image: Image.Image, heatmap_b64: str, size):
    heatmap_img = Image.open(BytesIO(base64.b64decode(heatmap_b64))).convert("RGBA")
    base_resized = base_image.convert("RGBA").resize(size)
    heatmap_resized = heatmap_img.resize(size)
    blended = Image.alpha_composite(base_resized, heatmap_resized)
    return heatmap_img, blended


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def unique_path(path: Path) -> Path:
    """Return a unique path by appending _1, _2, ... if needed."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def process_benchmark(args):
    if not load_data():
        return

    patch_indices = build_image_patch_dict(decoded_tokens)
    assistant_text = tokens_to_text(decoded_tokens)

    benchmark_file = Path(args.benchmark_file)
    if not benchmark_file.exists():
        print(f"Benchmark file not found: {benchmark_file}")
        return

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    images_dir = output_dir / "images"
    ensure_output_dir(images_dir)

    summary = []
    predictions = []
    total = 0
    matched = 0
    processed = 0

    with benchmark_file.open() as f:
        for line in f:
            if args.limit and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            total += 1
            item = json.loads(line)
            qid = item.get("question_id", total)
            question_text = item.get("text", "")
            label = item.get("label", "").lower()
            image_name = item.get("image")

            main_word = extract_target_word(question_text)
            if not main_word:
                print(f"[{qid}] Could not extract target word from question: {question_text}")
                continue

            idx = find_token_index(main_word, decoded_tokens)
            predicted_label = "yes" if idx is not None else "no"
            is_correct = predicted_label == label
            matched += int(is_correct)

            heatmap_path = None
            overlay_path = None
            base_image_save_path = None

            # Resolve image path before building heatmap so we can size correctly
            image_path = None
            if image_name:
                candidate = Path(args.image_root) / image_name
                if candidate.exists():
                    image_path = candidate
            if image_path is None and Path("image.png").exists():
                image_path = Path("image.png")

            if image_path is not None:
                base_img = Image.open(image_path).convert("RGB")
            else:
                base_img = Image.new(
                    "RGB", (smolvlm_patch_size * 4, smolvlm_patch_size * 3), color="white"
                )

            base_image_save_path = unique_path(images_dir / f"{qid}_{main_word}_image.png")
            base_img.save(base_image_save_path)

            if idx is not None:
                word_attn, _ = get_word_attention(main_word, decoded_tokens, args.layer, idx)
                if word_attn is not None:
                    attn_map = get_attn_map(word_attn, args.head, patch_indices)
                else:
                    attn_map = np.zeros(
                        (smolvlm_patch_size * 3, smolvlm_patch_size * 4), dtype=np.float32
                    )
            else:
                attn_map = np.zeros(
                    (smolvlm_patch_size * 3, smolvlm_patch_size * 4), dtype=np.float32
                )

            heatmap_b64, _, resized_w, resized_h = create_attention_heatmap(
                attn_map, base_img, args.threshold
            )

            heatmap_img, blended = overlay_heatmap_on_image(
                base_img, heatmap_b64, (resized_w, resized_h)
            )

            heatmap_path = unique_path(images_dir / f"{qid}_{main_word}_heatmap.png")
            overlay_path = unique_path(images_dir / f"{qid}_{main_word}_overlay.png")
            heatmap_img.save(heatmap_path)
            blended.save(overlay_path)

            del attn_map, heatmap_img, blended, base_img
            gc.collect()

            summary.append(
                {
                    "question_id": qid,
                    "question": question_text,
                    "label": label,
                    "predicted_label": predicted_label,
                    "is_correct": is_correct,
                    "target_word": main_word,
                    "image": image_name,
                    "base_image_path": str(base_image_save_path) if base_image_save_path else None,
                    "heatmap_path": str(heatmap_path) if heatmap_path else None,
                    "overlay_path": str(overlay_path) if overlay_path else None,
                }
            )
            predictions.append({"question": question_text, "answer": predicted_label})
            processed += 1
            print(
                f"[{qid}] word='{main_word}' label={label} predicted={predicted_label} correct={is_correct}"
            )

    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w") as out:
        for row in summary:
            out.write(json.dumps(row) + "\n")

    print(f"\nProcessed {processed} items (requested limit: {args.limit or 'all'}).")
    print(f"Accuracy: {matched}/{processed} = {matched/processed if processed else 0:.3f}")
    print(f"Summary saved to {summary_path}")
    print(f"Heatmaps/overlays saved under {output_dir}")

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w") as out:
        for row in predictions:
            out.write(json.dumps(row) + "\n")
    print(f"Predictions (question/answer) saved to {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline benchmark runner for attention heatmaps.")
    parser.add_argument(
        "--benchmark-file",
        default="../POPE/output/coco/coco_pope_popular_100.json",
        help="Path to POPE-style JSONL benchmark file.",
    )
    parser.add_argument(
        "--image-root",
        default="../COCO/val2014",
        help="Root directory containing benchmark images (image field is appended).",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_output",
        help="Directory to store outputs (heatmaps, overlays, summary).",
    )
    parser.add_argument("--layer", type=int, default=20, help="Layer index for attention.")
    parser.add_argument("--head", type=int, default=5, help="Head index for attention.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum attention threshold applied before normalization.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only first N samples (0 = all)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Reduce matplotlib warnings in CLI mode
    matplotlib.rcParams["figure.max_open_warning"] = 0
    plt.ioff()
    args = parse_args()
    process_benchmark(args)
