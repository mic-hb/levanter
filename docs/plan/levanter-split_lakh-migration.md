---
name: Levanter split_lakh Migration
overview: Migrate the Levanter split_lakh branch codebase (41 Python source files across src/levanter/ and src/haliax/) from deprecated JAX ~0.3/0.4 + Equinox 0.9 APIs to a modern JAX 0.4.30+ / Equinox 0.11+ stack, enabling fine-tuning of the Anticipatory Music Transformer from the converted HF checkpoint.
todos:
  - id: phase0-setup
    content: "Phase 0: Create migration branch, write new requirements.txt with modern pins, install deps into .venv"
    status: completed
  - id: phase1-jax-imports
    content: "Phase 1: Fix all 12 files with deprecated JAX API imports (PartitionSpec, Mesh, GlobalDeviceArray, pjit, gda_serialization, etc.)"
    status: completed
  - id: phase1-critical-rewrites
    content: "Phase 1 (critical rewrites): Rewrite named_pjit (partitioning.py), global_key_array (jax_utils.py), GlobalBatchDataset.__iter__ (sharded.py) to use modern jax.jit / jax.make_array_from_callback"
    status: completed
  - id: phase2-equinox
    content: "Phase 2: Fix 3 files with Equinox internal API imports (compile_utils, module.Static, custom_types.BoolAxisSpec)"
    status: completed
  - id: phase3-hf-checkpoints
    content: "Phase 3: Migrate hf_checkpoints.py from cached_download to hf_hub_download; verify checkpoint loading"
    status: completed
  - id: phase4-tests
    content: "Phase 4: Fix test imports (test_global_batch_dataset.py, test_partitioning.py)"
    status: completed
  - id: phase5-config
    content: "Phase 5: Create local fine-tuning config YAML, prepare tokenized training data"
    status: completed
  - id: phase6-validation
    content: "Phase 6: Run all 5 validation tests (import smoke, CLI help, checkpoint load, 1-step train, unit tests)"
    status: completed
isProject: false
---

# Levanter split_lakh Branch -- Modern JAX/Equinox Migration Plan

## Problem Statement

The `lib/levanter-split_lakh` codebase was written against JAX ~0.3-0.4.1 and Equinox ~0.9.0 (circa late 2022). Multiple APIs it depends on have been fully removed in modern JAX (0.4.30+) and Equinox (0.11+):

- `jax.experimental.global_device_array.GlobalDeviceArray` (removed JAX 0.4.7)
- `jax.experimental.pjit.pjit / FROM_GDA / with_sharding_constraint` (deprecated, use `jax.jit`)
- `jax.interpreters.pxla.PartitionSpec / Mesh / ShardedDeviceArray` (internal, removed)
- `jax.experimental.maps.Mesh / thread_resources` (removed)
- `jax.experimental.gda_serialization` (renamed to `array_serialization`)
- `jax.random.KeyArray` type annotation (removed)
- `jax.config.update("jax_array", ...)` (no-op, always true now)
- `jax._src.clusters.SlurmCluster / TpuCluster` (internal, may be removed)
- `equinox.serialisation._is_index / default_deserialise_filter_spec / default_serialise_filter_spec` (private API)
- `equinox.compile_utils.compile_cache / get_fun_names / hashable_combine / hashable_partition` (private internal API)
- `equinox.module.Static` (internal)
- `equinox.custom_types.BoolAxisSpec` (internal)
- `huggingface_hub.cached_download / hf_hub_url` (removed in HF Hub 0.26)

These cannot be resolved by pinning old versions because matching `jaxlib` wheels are no longer published for the current platform.

---

## Target Stack

- **Python**: 3.10 (matches `pyproject.toml`)
- **JAX**: 0.4.30 (latest stable with CPU wheels for linux/x86_64)
- **jaxlib**: 0.4.30
- **Equinox**: 0.11.x (latest compatible with JAX 0.4.30)
- **Optax**: 0.2.x
- **Chex**: 0.1.x (latest compatible)
- **NumPy**: >=1.24, <2.0 (avoid NumPy 2 ABI breaks)
- **Transformers**: >=4.41
- **huggingface_hub**: >=0.23 (has `hf_hub_download`)
- **safetensors**: >=0.4
- **PyTorch**: >=2.0 (for checkpoint loading only)
- **jmp**: latest from git main
- **Other**: wandb, pyrallis, pyarrow, braceexpand, fsspec, tensorstore, pytimeparse, humanfriendly, gitpython, tqdm, datasets

---

## Phase 0: Pre-Migration Setup

### Step 0.1 -- Create a clean branch for migration work

Create a new git branch from the `split_lakh` worktree to track all migration changes independently.

**AC**: A branch like `split_lakh-modern-jax` exists and is checked out in the worktree.

### Step 0.2 -- Write the new requirements.txt

Replace [requirements.txt](lib/levanter-split_lakh/requirements.txt) with pinned modern versions:

```
jax==0.4.30
jaxlib==0.4.30
equinox==0.11.9
optax==0.2.4
chex==0.1.88
numpy>=1.24,<2
jaxtyping>=0.2.25
transformers>=4.41
huggingface_hub>=0.23
safetensors>=0.4
torch>=2.0
jmp @ git+https://github.com/deepmind/jmp@main
wandb
pyrallis
pyarrow
braceexpand
fsspec>=2023.1
tensorstore>=0.1.45
pytimeparse
humanfriendly
gitpython
tqdm
datasets>=2.14
```

**AC**: `uv pip install -r requirements.txt` succeeds without conflicts. `python -c "import jax; print(jax.__version__)"` prints `0.4.30`.

### Step 0.3 -- Install levanter in editable mode

```bash
cd lib/levanter-split_lakh
source .venv/bin/activate
pip install -e .
```

**AC**: `python -c "import levanter"` runs (will likely fail at this point due to import errors -- that is expected and Phase 1 will fix it).

---

## Phase 1: Fix JAX API Imports (12 files)

Each file below has one or more deprecated JAX imports. The table shows every import that must change. After each file is patched, run `python -c "import <module>"` to verify.

### 1.1 -- [src/haliax/partitioning.py](lib/levanter-split_lakh/src/haliax/partitioning.py) (CRITICAL -- most complex)

This is the most heavily impacted file. It uses `GlobalDeviceArray`, `FROM_GDA`, `pjit`, `with_sharding_constraint`, and `PartitionSpec` all from removed modules. The `named_pjit` function is the core abstraction used throughout the training loop.

**Deprecated imports (lines 12-14, 288):**

```python
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.pjit import FROM_GDA, pjit, with_sharding_constraint
from jax.interpreters.pxla import PartitionSpec
...
from jax.experimental.maps import thread_resources
```

**Modern replacements:**

```python
from jax.sharding import PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
# pjit -> jax.jit
# GlobalDeviceArray -> jax.Array (no import needed)
# FROM_GDA -> removed entirely (no longer needed; jax.jit handles jax.Array natively)
```

**Code changes required:**

- `infer_resource_partitions` (line 134): Remove the `isinstance(node, GlobalDeviceArray)` branch and the `FROM_GDA` return. Modern `jax.Array` objects carry their own sharding; return `node.sharding` instead.

- `named_pjit` function (lines 144-223): The core wrapper around `pjit`. Replace `pjit(...)` call at line 257 with `jax.jit(...)` and rename keyword args from `in_axis_resources`/`out_axis_resources` to `in_shardings`/`out_shardings`.

- `physical_axis_size` (line 288): Replace `from jax.experimental.maps import thread_resources` with the mesh context available from `jax.sharding.Mesh`. The modern approach is to accept the mesh as a parameter or read it from a context variable rather than using `thread_resources.env`.

**AC**: `python -c "from haliax.partitioning import named_pjit, PartitionSpec"` succeeds.

### 1.2 -- [src/haliax/jax_utils.py](lib/levanter-split_lakh/src/haliax/jax_utils.py)

**Deprecated (line 8, 14):**

```python
from equinox.module import Static
...
def shaped_rng_split(key, ...) -> jrandom.KeyArray:
```

**Replacement:**

```python
from equinox import Module as _EqxModule  # Static may be at equinox._module.Static or equinox.internal
# For KeyArray: just use jax.Array or remove the return type annotation
```

`equinox.module.Static` was an internal class. In modern Equinox it may be at `equinox.internal._module.Static` or simply `equinox.nn.StaticInt` / similar. The simplest fix: since `Static` is only used in `filter_eval_shape` to wrap static outputs, replace with `eqx.internal.static_field` or a simple wrapper dataclass.

For `KeyArray`: change `-> jrandom.KeyArray` to `-> jax.Array`.

**AC**: `python -c "from haliax.jax_utils import shaped_rng_split, filter_eval_shape"` succeeds.

### 1.3 -- [src/haliax/hof.py](lib/levanter-split_lakh/src/haliax/hof.py) (line 9)

**Deprecated:**

```python
from equinox.custom_types import BoolAxisSpec
```

**Replacement:** In modern Equinox, `BoolAxisSpec` is likely at `equinox.nn._shared.BoolAxisSpec` or the type can be replaced with a simple `Union[bool, Callable]` annotation. Check `equinox` source or use `Any`.

**AC**: `python -c "from haliax.hof import scan"` succeeds.

### 1.4 -- [src/levanter/jax_utils.py](lib/levanter-split_lakh/src/levanter/jax_utils.py)

**Deprecated (lines 8, 12-13):**

```python
from chex import PRNGKey
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.interpreters.pxla import PartitionSpec
```

**Replacement:**

```python
from jax import Array as JaxArray  # or just use jax.Array inline
from jax.sharding import PartitionSpec, NamedSharding
```

**Key code change -- `global_key_array` (lines 82-119):** Replace `GlobalDeviceArray.from_callback(...)` with `jax.make_array_from_callback(...)`. The API difference:

```python
# OLD:
GlobalDeviceArray.from_callback(global_shape, global_mesh, mesh_axes, data_callback)
# NEW:
jax.make_array_from_callback(global_shape, NamedSharding(global_mesh, mesh_axes), data_callback)
```

Note the commented-out code at lines 109-112 already shows the modern pattern -- uncomment and use it.

Also: `flops_estimate` (line 37) uses `jax.xla_computation` which is deprecated in favor of `jax.jit(fn).lower(*args).compile().cost_analysis()`.

**AC**: `python -c "from levanter.jax_utils import global_key_array, parameter_count"` succeeds.

### 1.5 -- [src/levanter/data/sharded.py](lib/levanter-split_lakh/src/levanter/data/sharded.py)

**Deprecated (lines 8-10):**

```python
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.multihost_utils import process_allgather
from jax.interpreters.pxla import Mesh, PartitionSpec
```

**Replacement:**

```python
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.multihost_utils import process_allgather  # still available
```

**Key code change -- `GlobalBatchDataset.__iter__` (lines 125-138):** Replace `GlobalDeviceArray.from_callback(...)` with `jax.make_array_from_callback(...)`. Again the commented-out modern code at lines 126-130 shows the pattern.

**AC**: `python -c "from levanter.data.sharded import GlobalBatchDataset"` succeeds.

### 1.6 -- [src/levanter/grad_accum.py](lib/levanter-split_lakh/src/levanter/grad_accum.py)

**Deprecated (lines 5-6):**

```python
from jax.experimental.pjit import with_sharding_constraint
from jax.interpreters.pxla import PartitionSpec
```

**Replacement:**

```python
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
```

**AC**: `python -c "from levanter.grad_accum import accumulate_gradients_sharded"` succeeds.

### 1.7 -- [src/levanter/config.py](lib/levanter-split_lakh/src/levanter/config.py)

**Deprecated (lines 18-19, 273, 329):**

```python
from jax._src.clusters import SlurmCluster, TpuCluster
from jax.experimental.maps import Mesh
...
use_jax_array: bool = True
...
jax.config.update("jax_array", self.use_jax_array)
```

**Replacement:**

```python
from jax.sharding import Mesh
# SlurmCluster/TpuCluster: wrap in try/except or remove if not needed for local training
```

- Remove the `use_jax_array` field and the `jax.config.update("jax_array", ...)` call entirely -- `jax.Array` is always enabled in modern JAX.

**AC**: `python -c "from levanter.config import TrainerConfig"` succeeds.

### 1.8 -- [src/levanter/mesh.py](lib/levanter-split_lakh/src/levanter/mesh.py) (line 5)

```python
# OLD:
from jax.experimental.maps import Mesh
# NEW:
from jax.sharding import Mesh
```

**AC**: `python -c "from levanter.mesh import process_mesh_position"` succeeds.

### 1.9 -- [src/levanter/tensorstore_serialization.py](lib/levanter-split_lakh/src/levanter/tensorstore_serialization.py)

**Deprecated (lines 8, 13):**

```python
import jax.experimental.gda_serialization.serialization as gda_ser
from jax.interpreters.pxla import ShardedDeviceArray
```

**Replacement:**

```python
import jax.experimental.array_serialization.serialization as array_ser
# ShardedDeviceArray -> jax.Array (unified)
```

Replace all `gda_ser.` references with `array_ser.` (the API surface -- `async_serialize`, `async_deserialize`, `get_tensorstore_spec`, `TS_CONTEXT` -- is the same, just renamed module). Remove the `ShardedDeviceArray` assertion at line 63 (all arrays are `jax.Array` now).

**AC**: `python -c "from levanter.tensorstore_serialization import tree_serialize_leaves_tensorstore"` succeeds.

### 1.10 -- [src/levanter/checkpoint.py](lib/levanter-split_lakh/src/levanter/checkpoint.py) (line 12)

**Deprecated:**

```python
from equinox.serialisation import _is_index, default_deserialise_filter_spec, default_serialise_filter_spec
```

**Replacement:** In modern Equinox, these are at the top level:

```python
from equinox import default_serialise_filter_spec, default_deserialise_filter_spec
```

For `_is_index`: this is a private helper. Check if `equinox` still exposes it internally (`equinox._serialisation._is_index`). If not, replicate its simple logic (it just checks if a value is a `jnp.ndarray` or `np.ndarray`).

**AC**: `python -c "from levanter.checkpoint import Checkpointer"` succeeds.

### 1.11 -- [src/levanter/models/longformer_scale_test.py](lib/levanter-split_lakh/src/levanter/models/longformer_scale_test.py) (line 5)

```python
# OLD:
from jax.experimental.maps import Mesh
# NEW:
from jax.sharding import Mesh
```

### 1.12 -- [src/levanter/compat/hf_checkpoints.py](lib/levanter-split_lakh/src/levanter/compat/hf_checkpoints.py) (line 5)

**Deprecated:**

```python
from huggingface_hub import cached_download, hf_hub_url
```

**Replacement:**

```python
from huggingface_hub import hf_hub_download
```

**Code changes:**

- `load_hf_model_checkpoint` (lines 24-29): Replace `cached_download(hf_hub_url(repo_id, filename))` with `hf_hub_download(repo_id, filename, revision=revision)`.

**AC**: `python -c "from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint"` succeeds.

---

## Phase 2: Fix Equinox Internal API Usage (3 files)

### 2.1 -- Equinox `compile_utils` in [src/haliax/partitioning.py](lib/levanter-split_lakh/src/haliax/partitioning.py) (line 11)

```python
from equinox.compile_utils import compile_cache, get_fun_names, hashable_combine, hashable_partition
```

These are internal Equinox utilities for caching JIT-compiled functions. In modern Equinox they have moved to `equinox.internal` or `equinox._compile_utils`. Strategy:

1. First check: `python -c "from equinox.compile_utils import compile_cache"` -- if it works, no change needed.
2. If not, try: `from equinox.internal._compile_utils import compile_cache, ...`
3. If that also fails: the `named_pjit` function that uses these should be rewritten to use `jax.jit` with `eqx.filter_jit` directly, which is simpler and more maintainable.

### 2.2 -- Equinox `module.Static` in [src/haliax/jax_utils.py](lib/levanter-split_lakh/src/haliax/jax_utils.py) (line 8)

Same strategy as above. Try `equinox.internal._module.Static` or replace usage with a simple frozen dataclass wrapper.

### 2.3 -- Equinox `custom_types.BoolAxisSpec` in [src/haliax/hof.py](lib/levanter-split_lakh/src/haliax/hof.py) (line 9)

Try `equinox.internal._custom_types.BoolAxisSpec` or replace the type hint with `Any`.

---

## Phase 3: Fix Training Script ([examples/gpt2_example.py](lib/levanter-split_lakh/examples/gpt2_example.py))

### 3.1 -- Update PartitionSpec import (already done via try/except at line 10-13)

Already handled -- no change needed.

### 3.2 -- Verify `named_pjit` usage works after Phase 1 changes

The training script calls `named_pjit` extensively. After Phase 1.1 patches `named_pjit` to use `jax.jit`, verify the decorators at lines 111, 122, 174, 199 still work.

### 3.3 -- Verify `global_key_array` usage (line 278)

After Phase 1.4, confirm the call at line 278 works with the new `jax.make_array_from_callback` implementation.

### 3.4 -- Verify HF checkpoint loading (line 120)

After Phase 1.12, confirm `load_hf_gpt2_checkpoint` can load the converted checkpoint at `artifacts/hf-converted/music-large-800k-bin/`.

**AC**: `PYTHONPATH=src python examples/gpt2_example.py --help` prints the help message without errors, including the `initialize_from_hf_checkpoint` argument.

---

## Phase 4: Fix Test Files (2 files)

### 4.1 -- [tests/test_global_batch_dataset.py](lib/levanter-split_lakh/tests/test_global_batch_dataset.py)

```python
# OLD:
from jax.experimental.global_device_array import Shard
from jax.experimental.maps import Mesh
# NEW:
from jax.sharding import Mesh
# Shard -> not needed or use jax.Shard if available
```

### 4.2 -- [tests/haliax/test_partitioning.py](lib/levanter-split_lakh/tests/haliax/test_partitioning.py)

```python
# OLD:
from jax.interpreters.pxla import PartitionSpec
# NEW:
from jax.sharding import PartitionSpec
```

**AC**: Tests pass with `PYTHONPATH=src python -m pytest tests/ -x`.

---

## Phase 5: Update Config Files for Local Training

### 5.1 -- Create a local fine-tuning config

The existing configs (e.g., [config/lakh_large_aar.yaml](lib/levanter-split_lakh/config/lakh_large_aar.yaml)) point to `gs://` Google Cloud Storage URLs. Create a local config `config/lakh_local_finetune.yaml` that:

- Points `train_urls` and `validation_urls` to local tokenized data files
- Points `checkpointer.base_path` to a local directory
- Sets conservative `num_train_steps` and `per_device_parallelism` for initial testing

### 5.2 -- Prepare tokenized training data

The training data must be pre-tokenized using the AMT tokenizer (passthrough mode). The existing pipeline in [src/levanter/data/text.py](lib/levanter-split_lakh/src/levanter/data/text.py) handles this via `CachedLMDatasetConfig` with `tokenizer: "passthrough"` and `plaintext: True`. The data files are plain text where each line is a space-separated sequence of integer tokens.

Follow the instructions in [lib/anticipation/train/README.md](lib/anticipation/train/README.md) to preprocess MIDI files into tokenized training data.

**AC**: Local config file exists. `train_urls` point to real, existing tokenized data files.

---

## Phase 6: Validation and End-to-End Testing

### Test 1: Import Smoke Test

```bash
cd lib/levanter-split_lakh && source .venv/bin/activate
PYTHONPATH=src python -c "
import jax; print('JAX', jax.__version__)
import equinox; print('Equinox', equinox.__version__)
import optax; print('Optax', optax.__version__)
import levanter; print('Levanter OK')
import haliax; print('Haliax OK')
from haliax.partitioning import named_pjit
from levanter.data.sharded import GlobalBatchDataset
from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint
print('All critical imports OK')
"
```

**AC**: All imports succeed, no tracebacks.

### Test 2: CLI Help

```bash
PYTHONPATH=src python examples/gpt2_example.py --help
```

**AC**: Help text prints without errors, shows all config fields including `initialize_from_hf_checkpoint`.

### Test 3: HF Checkpoint Loading

```bash
PYTHONPATH=src python -c "
from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint
model = load_hf_gpt2_checkpoint('artifacts/hf-converted/music-large-800k-bin')
print('Model loaded successfully')
print('Params:', sum(p.size for p in jax.tree_util.tree_leaves(model) if hasattr(p, 'size')))
"
```

**AC**: Model loads without errors. Parameter count matches expected (~86M for music-large-800k).

### Test 4: One-Step Training Run

```bash
PYTHONPATH=src python examples/gpt2_example.py \
  --config_path config/lakh_local_finetune.yaml \
  --initialize_from_hf_checkpoint artifacts/hf-converted/music-large-800k-bin \
  --trainer.num_train_steps 1 \
  --trainer.wandb.mode disabled
```

**AC**: Completes one training step without crashing. Prints a loss value.

### Test 5: Unit Tests

```bash
PYTHONPATH=src python -m pytest tests/ -x -v
```

**AC**: All tests pass or only fail for reasons unrelated to the migration (e.g., missing GCS data).

---

## Complete File Impact Matrix

| File                                           | Changes                                                                                     | Risk   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- | ------ |
| `src/haliax/partitioning.py`                   | 6 import changes, rewrite `named_pjit` + `infer_resource_partitions` + `physical_axis_size` | HIGH   |
| `src/haliax/jax_utils.py`                      | 2 import changes (`Static`, `KeyArray`)                                                     | MEDIUM |
| `src/haliax/hof.py`                            | 1 import change (`BoolAxisSpec`)                                                            | LOW    |
| `src/levanter/jax_utils.py`                    | 3 import changes, rewrite `global_key_array`, update `flops_estimate`                       | HIGH   |
| `src/levanter/data/sharded.py`                 | 3 import changes, rewrite `GlobalBatchDataset.__iter__`                                     | HIGH   |
| `src/levanter/grad_accum.py`                   | 2 import changes                                                                            | LOW    |
| `src/levanter/config.py`                       | 3 import changes, remove `use_jax_array` field + config call                                | MEDIUM |
| `src/levanter/mesh.py`                         | 1 import change                                                                             | LOW    |
| `src/levanter/tensorstore_serialization.py`    | 2 import changes, rename module ref, remove `ShardedDeviceArray` assert                     | MEDIUM |
| `src/levanter/checkpoint.py`                   | 1 import change (Equinox serialization internals)                                           | MEDIUM |
| `src/levanter/models/longformer_scale_test.py` | 1 import change                                                                             | LOW    |
| `src/levanter/compat/hf_checkpoints.py`        | 1 import change, rewrite download logic                                                     | LOW    |
| `examples/gpt2_example.py`                     | Already partially patched; verify after upstream changes                                    | LOW    |
| `tests/test_global_batch_dataset.py`           | 2 import changes                                                                            | LOW    |
| `tests/haliax/test_partitioning.py`            | 1 import change                                                                             | LOW    |
| `requirements.txt`                             | Full rewrite with modern pins                                                               | LOW    |

---

## Key References

- **JAX Array Migration Guide**: https://docs.jax.dev/en/latest/jax_array_migration.html
- **JAX Changelog**: https://github.com/jax-ml/jax/blob/main/CHANGELOG.md
- **Equinox Serialization Docs**: https://docs.kidger.site/equinox/api/serialisation/
- **Equinox GitHub**: https://github.com/patrick-kidger/equinox
- **HuggingFace Hub Migration**: https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/concepts/migration.md
- **`jax.make_array_from_callback` Docs**: https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_callback.html
- **pjit to jit Migration**: `in_axis_resources` -> `in_shardings`, `out_axis_resources` -> `out_shardings`
- **AMT Training README**: [lib/anticipation/train/README.md](lib/anticipation/train/README.md)
- **Converted Checkpoint**: [lib/levanter-split_lakh/artifacts/hf-converted/music-large-800k-bin/](lib/levanter-split_lakh/artifacts/hf-converted/music-large-800k-bin/)
