# Video Bench

Generate text-to-video benchmark datasets with local Hugging Face Diffusers
models.

## Download and cache models

Install the environment and authenticate with Hugging Face if the model is
gated:

```bash
UV_CACHE_DIR=/tmp/$USER/uv-cache uv sync --extra dev
uv run hf auth login
```

In [`config.yaml`](config.yaml), enable the models you want and set a shared
cache directory:

```yaml
runtime:
  model_cache_dir: /cluster/scratch/nsoldati/video-generation/huggingface
  local_files_only: true
```

Download every enabled model on the login node. The checkpoint name is in its
file under [`configs/`](configs):

```bash
uv run hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir /cluster/scratch/nsoldati/video-generation/huggingface
```

It is recommended to run downloads in tmux sessions, expecially for large models.

Using the same cache path with `local_files_only: true` lets compute nodes load
the model without internet access.

## Run generation

Set the prompt file, seeds, output directory, and enabled models in
[`config.yaml`](config.yaml), then validate the job:

```bash
uv run ./generate.py --dry-run
```

Run on one GPU:

```bash
srun --gpus=1 --time=04:00:00 uv run --no-sync ./generate.py
```

Submit the same run as a detached SLURM batch job (output is written to
`slurm-<job-id>.out`):

```bash
sbatch --job-name=video-bench --gpus=pro_6000:1 --time=14:00:00 \
  --output=slurm-%j.out --wrap='uv run --no-sync ./generate.py'
```

Completed videos are skipped when a run is restarted. Results are written to
the configured output directory with a `manifest.csv` and a `videos/` folder.
