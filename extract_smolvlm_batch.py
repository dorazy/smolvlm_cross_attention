"""
Batch extraction of decoded_tokens.npy and attention_array.npy per image using
HuggingFaceTB/SmolVLM-Instruct-256M.

Output layout per image (under --output-root/<image_stem>/):
  decoded_tokens.npy                  # numpy object array of tokens
  attention_array.npy                 # float16, shape (layers, 1, heads, seq_len, seq_len)
  attentions_5/attention_fp16_rounded_layer_<i>.npz  # per-layer attention, same dtype/shape as existing files

This aligns with the shapes found in the repo:
  - attention_fp16_rounded_layer_0.npz: (1, 9, 1125, 1125) float16
  - attention_fp16_rounded.npz: (30, 1, 9, 1125, 1125) float16
The script preserves (batch=1, heads, seq_len, seq_len) per layer and stacks into (layers, 1, heads, seq_len, seq_len).

Example:
  python extract_smolvlm_batch.py \
    --image-dir ../COCO/val2014 \
    --output-root ./smol_outputs \
    --prompt "Can you describe this image?"

Notes:
  - Requires internet on first run to download the model from HF (model id is configurable).
  - VRAM usage depends on image size and sequence length; set --device cpu if GPU is unavailable.
  - For POPE subsets, you can pass --limit to process the first N images.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser(description="Extract SmolVLM attentions/tokens per image.")
    p.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images (e.g., COCO/val2014). Processes all *.jpg|*.png.",
    )
    p.add_argument(
        "--output-root",
        default="smol_outputs",
        help="Root directory to write per-image outputs.",
    )
    p.add_argument(
        "--prompt",
        default="Can you describe this image?",
        help="Prompt text paired with each image.",
    )
    p.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM-Instruct-256M",
        help="Model id or local path.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (cuda, cuda:0, cpu). Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N images (0 = all).",
    )
    return p.parse_args()


def list_images(image_dir: Path, limit: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if limit and limit > 0:
        imgs = imgs[:limit]
    return imgs


def save_outputs(image_path: Path, tokens: List[str], attn_stack: torch.Tensor, output_root: Path):
    stem_dir = output_root / image_path.stem
    attn_dir = stem_dir / "attentions_5"
    stem_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(parents=True, exist_ok=True)

    # Save decoded_tokens.npy
    np.save(stem_dir / "decoded_tokens.npy", np.array(tokens, dtype=object))

    # Save attention_array.npy with shape (layers, 1, heads, seq, seq)
    np.save(stem_dir / "attention_array.npy", attn_stack.numpy().astype(np.float16))

    # Save per-layer .npz
    for i, layer_attn in enumerate(attn_stack):
        np.savez(attn_dir / f"attention_fp16_rounded_layer_{i}.npz", attention=layer_attn.numpy().astype(np.float16))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = Path(args.image_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir, args.limit)
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Loading model {args.model_id} on {device}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto" if device.startswith("cuda") else None,
    ).to(device)

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat_text, images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Tokens aligned to input_ids used for attentions
        token_ids = inputs["input_ids"][0].tolist()
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

        # outputs.attentions: list of length num_layers; each (batch=1, num_heads, seq, seq)
        attn_tensors = [a.detach().cpu().to(torch.float16) for a in outputs.attentions]
        attn_stack = torch.stack(attn_tensors)  # (layers, 1, heads, seq, seq)

        save_outputs(img_path, tokens, attn_stack, output_root)

    print(f"Done. Outputs saved under {output_root}")


if __name__ == "__main__":
    main()
