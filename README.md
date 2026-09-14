# Video Bench

Video Bench generates reproducible benchmark datasets with local text-to-video
models. It uses Hugging Face Diffusers for inference and Accelerate to assign
independent prompt/seed jobs to one model replica per GPU.

Built-in adapters are included for:

- Wan (`WanPipeline`)
- HunyuanVideo (`HunyuanVideoPipeline`)
- CogVideoX (`CogVideoXPipeline`)
- LTX-Video (`LTXPipeline`)
- LTX-2 (`LTX2Pipeline`, including generated audio when available)

The generic `diffusers` adapter can load other compatible pipelines without a
code change. Model-specific behavior remains isolated in small adapter classes.

## Setup

Prerequisites are Python 3.10+, `uv`, a CUDA-capable PyTorch environment, and
enough temporary or scratch space for the selected checkpoints. The dependencies
include an FFmpeg binary for regular MP4 export and PyAV for muxing LTX-2 audio.

```bash
UV_CACHE_DIR=/tmp/$USER/uv-cache uv sync --extra dev
uv run accelerate config
```

If a checkpoint is gated, authenticate before running:

```bash
uv run hf auth login
```

The dependency set pins Diffusers to the tested `0.40.x` API line. Update the
pin deliberately when validating newer pipeline APIs.

## High-level configuration

[`config.yaml`](config.yaml) is the default input to `generate.py`. It selects
models, the prompt and seed counts, all output locations, and where downloaded
model files are cached:

```yaml
models:
  - config: configs/wan.yaml
    enabled: true
  - config: configs/cogvideox.yaml
    enabled: false

prompts:
  path: prompts.josn
  count: 10

seeds:
  start: 0
  count: 3

output:
  root_dir: output
  videos_dir: videos
  manifest: manifest.csv

runtime:
  model_cache_dir: /cluster/scratch/$USER/video-generation/huggingface
  local_files_only: true
  load_to_vram: true
```

Enabled models run sequentially, so only one model per worker needs to fit in
VRAM at a time. `load_to_vram: true` disables the CPU-offload option in the
individual model YAML files. Hugging Face checkpoints must still be downloaded
to a filesystem before loading. The default uses shared cluster scratch, never
this repository, so a checkpoint downloaded on the login node remains available
to compute nodes and future jobs. Since compute nodes have no outbound Hub
access, download each enabled checkpoint once from the login node. For the
default Wan model:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir /cluster/scratch/$USER/video-generation/huggingface
```

`local_files_only: true` then prevents network checks during the SLURM job.

## Run

Validate the complete job expansion without loading a model or writing output:

```bash
./generate.py --dry-run
```

After activating the prepared environment, a single-GPU SLURM job needs no CLI
arguments because `config.yaml` is the default:

```bash
srun --gpus=1 --time=04:00:00 ./generate.py
```

For multiple GPUs, launch one Accelerate process per GPU (replace `4` in both
places as needed):

```bash
srun --gpus=4 --time=04:00:00 \
  uv run --no-sync accelerate launch --num_processes 4 ./generate.py
```

Use `--config another.yaml` to select another suite, `--output-dir` to override
the root output directory, or `--overwrite` to regenerate completed videos.
`--prompts` and `--seeds` remain available as temporary CLI overrides.

Without `--overwrite`, a job is skipped only when its MP4 exists and its CSV row
has `status=completed`. Each successful row is written atomically as soon as its
video finishes, so rerunning the same command after a timeout resumes incomplete
work. An orphaned MP4 without a completed manifest row is regenerated.

All selected models share this layout:

```text
output/
├── manifest.csv
└── videos/
    └── wan-2.1-t2v-1.3b/
        └── 4d_001/
            ├── seed-0.mp4
            ├── seed-1.mp4
            └── seed-2.mp4
```

Each CSV row includes the status, model, adapter, checkpoint, prompt and prompt ID,
seed, resolution, frame count, FPS, inference steps, guidance scale, complete
generation parameters, relative output path, elapsed generation time, and any
source prompt metadata. A lock plus atomic file replacement prevents concurrent
GPU workers from corrupting the manifest. Failures are retained in
`.errors.<model>.rank-*.jsonl` and cause a non-zero exit after all distributed
workers finish their assigned jobs.

## Prompt format

[`prompts.josn`](prompts.josn) contains a sample. The preferred form is a list
of records:

```json
[
  {
    "id": "unique-stable-id",
    "prompt": "A detailed video caption...",
    "generation": {"guidance_scale": 4.5},
    "metadata": {
      "source_video": "real/videos/000123.mp4",
      "caption_model": "future-vlm-name",
      "split": "train"
    }
  }
]
```

`generation` optionally overrides generation settings for that prompt.
`metadata` is copied unchanged into `prompt_metadata` in the output CSV. This
is intended for a future VLM captioning stage: keep the stable ID, original
video path, caption provenance, and dataset split there so real/generated pairs
can be joined later. A plain JSON list of prompt strings and a top-level
`{"prompts": [...]}` object are also accepted.

## Model configuration

Ready-to-edit examples live in [`configs/`](configs). A config has three
sections:

```yaml
model:
  name: my-model                       # output directory and metadata label
  adapter: diffusers                   # or wan/hunyuan_video/cogvideox/ltx_video/ltx2
  checkpoint: org/checkpoint
  pipeline_class: diffusers.MochiPipeline  # required only for generic adapter
  dtype: bfloat16
  revision: null
  variant: null
  load_kwargs: {}

generation:
  width: 848
  height: 480
  num_frames: 81
  fps: 16
  num_inference_steps: 30
  guidance_scale: 5.0
  negative_prompt: null
  parameters: {}                      # passed directly to pipeline.__call__

runtime:
  output_dir: outputs                    # overridden by the high-level config
  model_cache_dir: null                  # overridden by the high-level config
  offload: model                      # null, model, or sequential
  vae_tiling: true
  vae_slicing: false
  attention_slicing: false
  torch_compile: false
  overwrite: false
```

`model.load_kwargs` is forwarded to `from_pretrained`. Components that need a
different dtype can be loaded explicitly. The Wan example loads its VAE in
float32 as recommended by the model documentation:

```yaml
components:
  vae:
    class: diffusers.AutoencoderKLWan
    dtype: float32
    subfolder: vae
```

Choose resolutions and frame counts supported by the checkpoint. For example,
Wan commonly requires frame counts of `4 * k + 1`. CPU offloading reduces VRAM
but is slower; set `offload: null` when a full model replica fits on each GPU.

## Adding a model

For a standard Diffusers text-to-video pipeline whose result exposes `.frames`,
use `adapter: diffusers` and set `model.pipeline_class`; no Python is needed.

For custom argument or export behavior, subclass
`DiffusersVideoGenerator` in
[`src/video_bench/generators/diffusers.py`](src/video_bench/generators/diffusers.py)
and register it with `register_generator` in
[`src/video_bench/generators/__init__.py`](src/video_bench/generators/__init__.py).
All adapters implement the common interface:

```python
generate(prompt, seed, output_path, **kwargs)
```

## Tests

The tests validate every supplied YAML config, supported prompt formats,
adapter argument handling, resumable CSV orchestration, and metadata merging
without downloading model weights:

```bash
UV_CACHE_DIR=/tmp/$USER/uv-cache uv run pytest
```
