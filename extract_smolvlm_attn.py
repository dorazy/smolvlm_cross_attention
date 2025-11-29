"""
Extract decoded_tokens.npy and attention_array.npy for a given image using
the SmolVLM Instruct 256M model from Hugging Face.

Example:
  python extract_smolvlm_attn.py \
    --image /home/jongmin/workspace/visvlm/COCO/val2014/COCO_val2014_000000310196.jpg \
    --output-dir . \
    --prompt "Can you describe this image?" \
    --model-id HuggingFaceTB/SmolVLM-Instruct-256M

Outputs:
  <output-dir>/decoded_tokens.npy        # token strings (dtype=object)
  <output-dir>/attention_array.npy       # shape (num_layers, num_heads, seq_len, seq_len) float16

Notes:
  - This script does NOT resize or preprocess images beyond what the model processor does.
  - Make sure you have enough VRAM; use --device cpu to force CPU (slow).
  - If you want per-image files, change --output-dir for each run or set --prefix.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Extract attention/tokens with SmolVLM Instruct 256M.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument(
        "--prompt",
        default="Can you describe this image?",
        help="Text prompt to pair with the image.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write decoded_tokens.npy and attention_array.npy.",
    )
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM-Instruct-256M",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g., cuda, cuda:0, cpu). Default chooses cuda if available.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix (e.g., image stem) for outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.prefix}_" if args.prefix else ""
    tokens_path = output_dir / f"{prefix}decoded_tokens.npy"
    attn_path = output_dir / f"{prefix}attention_array.npy"

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"Loading model: {args.model_id} on {device} (dtype={dtype})")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    ).to(device)
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    print(f"Reading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    print("Preparing inputs...")
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

    print("Running forward pass with attentions...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # tokens aligned with input_ids used for attentions
    token_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    np.save(tokens_path, np.array(tokens, dtype=object))
    print(f"Saved tokens to {tokens_path}")

    # outputs.attentions is a list (num_layers) of tensors (batch, num_heads, seq, seq)
    attn_tensors = [a[0].detach().cpu().to(torch.float16) for a in outputs.attentions]
    attn_stack = torch.stack(attn_tensors)  # (num_layers, num_heads, seq, seq)
    np.save(attn_path, attn_stack.numpy())
    print(f"Saved attention array to {attn_path} with shape {attn_stack.shape}")


if __name__ == "__main__":
    main()
