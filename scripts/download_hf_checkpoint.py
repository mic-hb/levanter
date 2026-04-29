"""
How to use this script:

Download a Hugging Face (HF) checkpoint and convert it to a Levanter checkpoint.

Example usage:
    python download_hf_checkpoint.py --source_model stanford-crfm/music-large-800k --output_dir artifacts/hf-converted/music-large-800k-bin

Arguments:
    --source_model    Source model name on Hugging Face (e.g. stanford-crfm/music-large-800k)
    --output_dir      Path to output directory for the converted checkpoint
"""

from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = ArgumentParser(description="Download a HF checkpoint and convert it to a Levanter checkpoint.")
    parser.add_argument("--source_model", default="stanford-crfm/music-large-800k", required=True, help="Source model name (e.g. stanford-crfm/music-large-800k)")
    parser.add_argument("--output_dir", default="artifacts/hf-converted/music-large-800k-bin", required=True, help="Path to output directory (e.g. artifacts/hf-converted/music-large-800k-bin)")
    return parser.parse_args()

def main():
    args = parse_args()

    source_model = args.source_model
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(source_model)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    # tokenizer = AutoTokenizer.from_pretrained(source_model)
    # tokenizer.save_pretrained(args.output_dir)
    print(f"Converted HF checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()