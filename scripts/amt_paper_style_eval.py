"""
Paper-style evaluator for Anticipatory Music Transformer checkpoints.

This script mirrors the metric definitions used in:
  lib/anticipation/scripts/eval-loss.py

In particular for arrival-time encoding:
  - token loss: mean next-token cross-entropy
  - ppl(e): exp(3 * token_loss)
  - ppl(t): exp(mean CE over token positions 0::3)
  - ppl(d): exp(mean CE over token positions 1::3)
  - ppl(n): exp(mean CE over token positions 2::3)

These are the same formulas referenced by the paper's Table 1 setup.
"""

import csv
import json
import math
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


EVENT_SIZE = 3


def parse_args():
    parser = ArgumentParser(description="Evaluate AMT checkpoint with paper-style metrics.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to HF-format checkpoint directory.")
    parser.add_argument("--dataset_path", required=True, help="Path to tokenized dataset txt (e.g. test.txt).")
    parser.add_argument(
        "--output_json",
        default="artifacts/eval-reports/amt-paper-style-eval.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output_csv",
        default="artifacts/eval-reports/amt-paper-style-eval.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Evaluate every N-th line. 1 means full dataset (default).",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=0,
        help="Optional cap on evaluated lines (0 means no cap).",
    )
    parser.add_argument(
        "--compute_bps",
        action="store_true",
        help="Compute bits-per-second using provided test-set hours.",
    )
    parser.add_argument(
        "--test_set_hours",
        type=float,
        default=560.98,
        help="Hours used for bps normalization (paper uses 560.98 for Lakh test).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for evaluation.",
    )
    return parser.parse_args()


def evaluate_checkpoint(model, dataset_path: str, subsample: int, max_lines: int):
    if subsample <= 0:
        raise ValueError("--subsample must be >= 1")

    ce_values = []
    num_lines_total = 0
    num_lines_used = 0
    start_time = time.time()

    with open(dataset_path, "r", encoding="utf-8") as data:
        for i, line in enumerate(data):
            num_lines_total += 1
            if i % subsample != 0:
                continue
            if max_lines > 0 and num_lines_used >= max_lines:
                break

            tokens = [int(token) for token in line.split()]
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor(tokens, dtype=torch.long, device=model.device).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_ids).logits[0]  # [seq, vocab]
                ce = F.cross_entropy(logits[:-1], input_ids[0, 1:], reduction="none")
                ce_values.append(ce.detach().cpu())

            num_lines_used += 1

    if len(ce_values) == 0:
        raise ValueError("No examples were evaluated. Check dataset path, subsample, and max_lines.")

    ce_all = torch.cat(ce_values, dim=0).numpy()
    elapsed = time.time() - start_time

    return {
        "ce_all": ce_all,
        "num_lines_total": num_lines_total,
        "num_lines_used": num_lines_used,
        "elapsed_seconds": elapsed,
    }


def compute_metrics(ce_all: np.ndarray, compute_bps: bool, subsample: int, test_set_hours: float):
    token_loss = float(np.mean(ce_all))
    result = {
        "token_loss": token_loss,
        "event_ppl": float(math.exp(EVENT_SIZE * token_loss)),
        "onset_ppl": float(math.exp(float(np.mean(ce_all[0::3])))),
        "dur_ppl": float(math.exp(float(np.mean(ce_all[1::3])))),
        "note_ppl": float(math.exp(float(np.mean(ce_all[2::3])))),
    }

    if compute_bps:
        # Same formula as anticipation/scripts/eval-loss.py
        # bps = subsample * ce_mean * log2(e) * (#tokens / (hours * 3600))
        bps = subsample * token_loss * math.log2(math.e) * (len(ce_all) / (test_set_hours * 3600.0))
        result["bps"] = float(bps)

    return result


def write_outputs(output_json: str, output_csv: str, payload: dict):
    output_json_path = Path(output_json)
    output_csv_path = Path(output_csv)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "checkpoint_path",
        "dataset_path",
        "subsample",
        "num_lines_used",
        "token_loss",
        "event_ppl",
        "onset_ppl",
        "dur_ppl",
        "note_ppl",
    ]
    if "bps" in payload["metrics"]:
        fields.append("bps")

    row = {
        "checkpoint_path": payload["checkpoint_path"],
        "dataset_path": payload["dataset_path"],
        "subsample": payload["subsample"],
        "num_lines_used": payload["dataset_stats"]["num_lines_used"],
        "token_loss": payload["metrics"]["token_loss"],
        "event_ppl": payload["metrics"]["event_ppl"],
        "onset_ppl": payload["metrics"]["onset_ppl"],
        "dur_ppl": payload["metrics"]["dur_ppl"],
        "note_ppl": payload["metrics"]["note_ppl"],
    }
    if "bps" in payload["metrics"]:
        row["bps"] = payload["metrics"]["bps"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path)
    model = model.to(args.device).eval()

    eval_result = evaluate_checkpoint(
        model=model,
        dataset_path=args.dataset_path,
        subsample=args.subsample,
        max_lines=args.max_lines,
    )
    metrics = compute_metrics(
        ce_all=eval_result["ce_all"],
        compute_bps=args.compute_bps,
        subsample=args.subsample,
        test_set_hours=args.test_set_hours,
    )

    payload = {
        "evaluator": "amt_paper_style_eval",
        "checkpoint_path": args.checkpoint_path,
        "dataset_path": args.dataset_path,
        "subsample": args.subsample,
        "max_lines": args.max_lines,
        "metrics": metrics,
        "dataset_stats": {
            "num_lines_total_seen": eval_result["num_lines_total"],
            "num_lines_used": eval_result["num_lines_used"],
            "num_token_predictions": int(len(eval_result["ce_all"])),
        },
        "runtime": {
            "elapsed_seconds": eval_result["elapsed_seconds"],
            "device": args.device,
        },
        "formula_reference": "Matches lib/anticipation/scripts/eval-loss.py formulas for arrival-time metrics.",
    }

    write_outputs(args.output_json, args.output_csv, payload)

    print("Paper-style evaluation complete.")
    print(f"token_loss: {metrics['token_loss']:.6f}")
    print(f"ppl(e): {metrics['event_ppl']:.6f}")
    print(f"ppl(t): {metrics['onset_ppl']:.6f}")
    print(f"ppl(d): {metrics['dur_ppl']:.6f}")
    print(f"ppl(n): {metrics['note_ppl']:.6f}")
    if "bps" in metrics:
        print(f"bps: {metrics['bps']:.6f}")
    print(f"JSON report: {args.output_json}")
    print(f"CSV report: {args.output_csv}")


if __name__ == "__main__":
    main()
