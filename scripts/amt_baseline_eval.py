import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import jax
import jmp
import pyrallis
from huggingface_hub.errors import EntryNotFoundError
from transformers import GPT2Config as HfGpt2Config

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_pjit
from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint
from levanter.config import TrainerConfig
from levanter.data.sharded import GlobalBatchDataset
from levanter.data.text import CachedLMDatasetConfig, TokenSeqDataset
from levanter.modeling_utils import cross_entropy_loss_and_log_normalizers
from levanter.models.gpt2 import Gpt2LMHeadModel


EXPECTED_LARGE_ARCH = {
    "n_layer": 36,
    "n_head": 20,
    "n_embd": 1280,
    "n_positions": 1024,
}


@dataclass
class BaselineEvalConfig:
    hf_checkpoint: str = "stanford-crfm/music-large-800k"
    hf_revision: Optional[str] = None
    valid_url: str = "../../anticipation/data/lmd_full/valid.txt"
    test_url: str = "../../anticipation/data/lmd_full/test.txt"
    report_path: str = "artifacts/eval-reports/music-large-800k-baseline.json"
    checkpoint_id: Optional[str] = None
    local_converted_checkpoint: Optional[str] = "artifacts/hf-converted/music-large-800k-bin"
    max_batches_per_split: Optional[int] = None
    log_every_n_batches: int = 100

    data: CachedLMDatasetConfig = field(default_factory=CachedLMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_json_safe(value):
    """
    Recursively coerce objects into JSON-serializable primitives.

    This avoids crashes from objects like jmp policy metadata or dtype meta objects
    that can appear inside nested dataclass configs.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    # Try common scalar-style conversions first.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    # Fallback to string representation for unknown config/meta objects.
    return str(value)


def _load_hf_architecture(hf_checkpoint: str, hf_revision: Optional[str]) -> Dict[str, int]:
    cfg = HfGpt2Config.from_pretrained(hf_checkpoint, revision=hf_revision)
    return {
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "n_positions": cfg.n_positions,
        "vocab_size": cfg.vocab_size,
    }


def _resolve_checkpoint_source(config: BaselineEvalConfig) -> str:
    """
    Resolve which checkpoint source to use for weight loading.

    Priority:
    1) Explicit local converted checkpoint path, when it exists and has pytorch_model.bin.
    2) hf_checkpoint (repo id or local path passed via --hf_checkpoint).
    """
    if config.local_converted_checkpoint:
        local_path = Path(config.local_converted_checkpoint).expanduser()
        if (local_path / "pytorch_model.bin").exists():
            return str(local_path)
    return config.hf_checkpoint


def _build_eval_dataset(
    base_data_config: CachedLMDatasetConfig,
    split_url: str,
    split_name: str,
    seq_len: int,
    trainer: TrainerConfig,
):
    split_data_config = copy.deepcopy(base_data_config)
    split_data_config.train_urls = []
    split_data_config.validation_urls = [split_url]
    split_data_config.cache_dir = str(Path(base_data_config.cache_dir) / f"baseline_eval_{split_name}")

    eval_batch = Axis("batch", trainer.eval_batch_size)
    return GlobalBatchDataset(
        TokenSeqDataset(split_data_config.build_or_load_document_cache("validation"), seq_len),
        trainer.device_mesh,
        eval_batch,
        trainer.compute_axis_mapping,
    )


def _evaluate_split(
    model: Gpt2LMHeadModel,
    dataset,
    trainer: TrainerConfig,
    *,
    split_name: str,
    max_batches_per_split: Optional[int],
    log_every_n_batches: int,
) -> Dict[str, float]:
    eval_batch = Axis("batch", trainer.eval_batch_size)
    seq_len = model.config.SeqLen
    key_seq_len = model.config.KeySeqLen
    vocab = model.Vocab
    mp: jmp.Policy = trainer.mp

    loss_mask = 1 - hax.nn.one_hot(-1, seq_len)

    def compute_loss(batch_input_ids):
        input_ids = hax.named(batch_input_ids, (eval_batch, seq_len))
        attn_mask = hax.nn.attention.causal_mask(seq_len, key_seq_len)
        logits = model(input_ids, attn_mask, key=None, inference=True)
        logits = mp.cast_to_output(logits)

        target_y = hax.roll(input_ids, -1, seq_len)
        target_y = hax.nn.one_hot(target_y, vocab, dtype=logits.dtype)

        token_loss, _ = cross_entropy_loss_and_log_normalizers(logits, vocab, target_y)
        return hax.mean(token_loss, where=loss_mask)

    eval_loss_pjit = named_pjit(compute_loss, axis_resources=trainer.parameter_axis_mapping)

    total_loss = 0.0
    num_batches = 0
    started_at = time.time()
    for batch in dataset:
        if max_batches_per_split is not None and num_batches >= max_batches_per_split:
            break
        scalar_loss = eval_loss_pjit(batch)
        total_loss += float(scalar_loss.item() if hasattr(scalar_loss, "item") else scalar_loss.scalar())
        num_batches += 1
        if log_every_n_batches > 0 and num_batches % log_every_n_batches == 0:
            elapsed = time.time() - started_at
            print(
                f"[{split_name}] batches={num_batches} "
                f"running_loss={total_loss / num_batches:.6f} "
                f"elapsed_s={elapsed:.1f}"
            )

    if num_batches == 0:
        raise ValueError("Evaluation dataset has zero batches.")

    avg_loss = total_loss / num_batches
    return {
        "eval_loss": avg_loss,
        "perplexity": float(math.exp(avg_loss)),
        "num_batches": float(num_batches),
        "elapsed_seconds": float(time.time() - started_at),
    }


@pyrallis.wrap()
def main(config: BaselineEvalConfig):
    # Levanter's named_pjit requires a non-empty logical->physical axis mapping.
    # Some CLI runs omit this, so provide the same safe defaults used by training configs.
    if not config.trainer.axis_resources:
        config.trainer.axis_resources = {
            "batch": "data",
            "vocab": "model",
            "mlp": "model",
            "heads": "model",
        }
    if not config.trainer.parameter_axis_resources:
        config.trainer.parameter_axis_resources = {
            "embed": "data",
        }

    if config.trainer.per_device_eval_parallelism == -1:
        config.trainer.per_device_eval_parallelism = max(1, config.trainer.per_device_parallelism)
    config.trainer.initialize(config)

    arch = _load_hf_architecture(config.hf_checkpoint, config.hf_revision)
    for key, expected in EXPECTED_LARGE_ARCH.items():
        if arch[key] != expected:
            raise ValueError(
                f"Checkpoint architecture mismatch for {key}: got {arch[key]}, expected {expected}. "
                f"Refusing to run because this script is for music-large-800k baseline only."
            )

    checkpoint_source = _resolve_checkpoint_source(config)

    with config.trainer.device_mesh:
        with jax.default_device(jax.devices("cpu")[0]):
            try:
                model = load_hf_gpt2_checkpoint(
                    checkpoint_source,
                    map_location="cpu",
                    revision=config.hf_revision if checkpoint_source == config.hf_checkpoint else None,
                )
            except EntryNotFoundError as exc:
                raise ValueError(
                    "Could not find pytorch_model.bin for the provided Hugging Face repo. "
                    "This is expected for safetensors-only repos like stanford-crfm/music-large-800k. "
                    "Use your converted local checkpoint directory instead, e.g. "
                    "--hf_checkpoint artifacts/hf-converted/music-large-800k-bin "
                    "or keep --local_converted_checkpoint pointing to that directory."
                ) from exc

        valid_dataset = _build_eval_dataset(
            base_data_config=config.data,
            split_url=config.valid_url,
            split_name="valid",
            seq_len=model.config.seq_len,
            trainer=config.trainer,
        )
        test_dataset = _build_eval_dataset(
            base_data_config=config.data,
            split_url=config.test_url,
            split_name="test",
            seq_len=model.config.seq_len,
            trainer=config.trainer,
        )

        started_at_unix = int(time.time())
        valid_metrics = _evaluate_split(
            model,
            valid_dataset,
            config.trainer,
            split_name="valid",
            max_batches_per_split=config.max_batches_per_split,
            log_every_n_batches=config.log_every_n_batches,
        )
        test_metrics = _evaluate_split(
            model,
            test_dataset,
            config.trainer,
            split_name="test",
            max_batches_per_split=config.max_batches_per_split,
            log_every_n_batches=config.log_every_n_batches,
        )

    resolved_config = _to_json_safe(asdict(config))
    resolved_config_checksum = _sha256_text(_canonical_json(resolved_config))

    report = {
        "run_type": "baseline_eval_only",
        "created_at_unix": started_at_unix,
        "checkpoint": {
            "id": config.checkpoint_id or config.hf_checkpoint,
            "path_or_repo": checkpoint_source,
            "revision": config.hf_revision if checkpoint_source == config.hf_checkpoint else None,
            "source_kind": "local_converted_dir" if checkpoint_source != config.hf_checkpoint else "hf_repo_or_local_path",
        },
        "architecture": arch,
        "metrics": {
            "valid": valid_metrics,
            "test": test_metrics,
        },
        "config": {
            "resolved": resolved_config,
            "checksum_sha256": resolved_config_checksum,
        },
        "notes": {
            "max_batches_per_split": config.max_batches_per_split,
            "log_every_n_batches": config.log_every_n_batches,
        },
    }

    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Baseline eval complete.")
    print(f"Checkpoint: {report['checkpoint']['path_or_repo']}")
    print(f"Valid loss/perplexity: {valid_metrics['eval_loss']:.6f} / {valid_metrics['perplexity']:.6f}")
    print(f"Test  loss/perplexity: {test_metrics['eval_loss']:.6f} / {test_metrics['perplexity']:.6f}")
    print(f"Report written to: {report_path}")
    print(f"Config checksum (sha256): {resolved_config_checksum}")


if __name__ == "__main__":
    main()
