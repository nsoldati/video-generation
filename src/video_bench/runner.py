"""Distributed benchmark generation orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from video_bench.config import BenchmarkConfig
from video_bench.generators import VideoGenerator, create_generator
from video_bench.metadata import append_jsonl, merge_metadata, read_csv, read_jsonl, upsert_csv
from video_bench.prompts import PromptRecord, slugify

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationJob:
    prompt: PromptRecord
    seed: int
    output_path: Path


def _make_jobs(
    prompts: list[PromptRecord], seeds: list[int], videos_root: Path,
) -> list[GenerationJob]:
    return [
        GenerationJob(prompt, seed, videos_root / prompt.id / f"seed-{seed}.mp4")
        for prompt in prompts
        for seed in seeds
    ]


def _parameters(config: BenchmarkConfig, prompt: PromptRecord) -> dict[str, Any]:
    params = config.generation.pipeline_parameters()
    protected = {"prompt", "generator"} & set(prompt.generation)
    if protected:
        raise ValueError(
            f"Prompt {prompt.id}: generation cannot override {', '.join(sorted(protected))}"
        )
    params.update(prompt.generation)
    _validate_parameters(prompt.id, params)
    return params


def _validate_parameters(prompt_id: str, params: dict[str, Any]) -> None:
    for key in ("width", "height", "num_frames", "num_inference_steps"):
        value = params.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Prompt {prompt_id}: generation.{key} must be a positive integer")
    for key in ("fps", "guidance_scale"):
        value = params.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Prompt {prompt_id}: generation.{key} must be a number")
    if params["fps"] <= 0:
        raise ValueError(f"Prompt {prompt_id}: generation.fps must be positive")


def _validate_seeds(seeds: list[int]) -> None:
    if not seeds:
        raise ValueError("At least one seed is required")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("Seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")


def run_generation(
    config: BenchmarkConfig,
    prompts: list[PromptRecord],
    seeds: list[int],
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    videos_dir: str | Path | None = None,
    overwrite: bool | None = None,
    dry_run: bool = False,
    generator_factory: Callable[[Any, Any, Any], VideoGenerator] = create_generator,
) -> Path:
    from accelerate import Accelerator
    import torch

    _validate_seeds(seeds)
    if not prompts:
        raise ValueError("At least one prompt is required")
    parameters_by_prompt = {prompt.id: _parameters(config, prompt) for prompt in prompts}

    accelerator = Accelerator()
    should_overwrite = config.runtime.overwrite if overwrite is None else overwrite
    root = Path(output_dir or config.runtime.output_dir).expanduser().resolve()
    csv_manifest = manifest_path is not None
    if csv_manifest:
        run_dir = root
        metadata_path = Path(manifest_path).expanduser().resolve()
        configured_videos = Path(videos_dir) if videos_dir is not None else Path("videos")
        if not configured_videos.is_absolute():
            configured_videos = root / configured_videos
        videos_root = configured_videos.resolve() / slugify(config.model.name)
        relative_root = root
    else:
        run_dir = root / slugify(config.model.name)
        metadata_path = run_dir / "metadata.jsonl"
        videos_root = run_dir / "videos"
        relative_root = run_dir
    jobs = _make_jobs(prompts, seeds, videos_root)

    if accelerator.is_main_process:
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "%s jobs for %s across %s process(es); output: %s",
            len(jobs), config.model.name, accelerator.num_processes, run_dir,
        )
    accelerator.wait_for_everyone()

    model_slug = slugify(config.model.name)
    shard = run_dir / f".metadata.rank-{accelerator.process_index}.jsonl"
    error_name = (
        f".errors.{model_slug}.rank-{accelerator.process_index}.jsonl"
        if csv_manifest else f".errors.rank-{accelerator.process_index}.jsonl"
    )
    error_shard = run_dir / error_name
    if not dry_run and not csv_manifest:
        shard.unlink(missing_ok=True)
    if not dry_run:
        error_shard.unlink(missing_ok=True)

    existing_records = read_csv(metadata_path) if csv_manifest else read_jsonl(metadata_path)
    recorded_outputs = {
        str(record.get("output_path"))
        for record in existing_records
        if isinstance(record.get("output_path"), str)
        and (not csv_manifest or record.get("status") == "completed")
    }

    local_errors = 0
    # Constructing an adapter does not load weights, but catches unknown adapters
    # and missing generic pipeline classes during a dry run.
    generator = generator_factory(config.model, config.runtime, accelerator.device)
    with accelerator.split_between_processes(jobs, apply_padding=False) as local_jobs:
        for job in local_jobs:
            relative_output = str(job.output_path.relative_to(relative_root))
            if (
                job.output_path.exists()
                and relative_output in recorded_outputs
                and not should_overwrite
            ):
                LOGGER.info("Skipping existing %s", job.output_path)
                continue
            started = time.perf_counter()
            try:
                params = parameters_by_prompt[job.prompt.id]
                if dry_run:
                    LOGGER.info("Would generate prompt=%s seed=%s", job.prompt.id, job.seed)
                    continue
                result = generator.generate(
                    job.prompt.prompt, job.seed, job.output_path, **params,
                )
                elapsed = time.perf_counter() - started
                record = {
                    "status": "completed",
                    "model": config.model.name,
                    "adapter": config.model.adapter,
                    "checkpoint": config.model.checkpoint,
                    "prompt_id": job.prompt.id,
                    "prompt": job.prompt.prompt,
                    "prompt_metadata": job.prompt.metadata,
                    "seed": job.seed,
                    "resolution": {"width": int(params["width"]), "height": int(params["height"])},
                    "frames": result.actual_num_frames or int(params["num_frames"]),
                    "fps": float(params["fps"]),
                    "inference_steps": int(params["num_inference_steps"]),
                    "guidance_scale": float(params["guidance_scale"]),
                    "generation_parameters": result.generation_parameters or params,
                    "output_path": relative_output,
                    "generation_seconds": round(elapsed, 3),
                }
                if csv_manifest:
                    upsert_csv(metadata_path, record)
                else:
                    append_jsonl(shard, record)
                LOGGER.info("Generated %s", job.output_path)
            except Exception as exc:  # Keep all ranks alive for clean distributed teardown.
                local_errors += 1
                append_jsonl(
                    error_shard,
                    {"prompt_id": job.prompt.id, "seed": job.seed,
                     "output_path": str(job.output_path),
                     "error": f"{type(exc).__name__}: {exc}"},
                )
                LOGGER.exception("Generation failed for prompt=%s seed=%s", job.prompt.id, job.seed)

    accelerator.wait_for_everyone()
    if dry_run:
        return metadata_path

    if accelerator.is_main_process and not csv_manifest:
        shards = [run_dir / f".metadata.rank-{rank}.jsonl" for rank in range(accelerator.num_processes)]
        # Overwrite controls MP4 regeneration; preserve metadata from other runs.
        # Records for regenerated paths are replaced by merge_metadata.
        merge_metadata(metadata_path, shards, overwrite=False)
    accelerator.wait_for_everyone()

    error_tensor = torch.tensor([local_errors], device=accelerator.device)
    total_errors = int(accelerator.reduce(error_tensor, reduction="sum").item())
    if total_errors:
        raise RuntimeError(
            f"{total_errors} generation job(s) failed. See {run_dir}/{error_name}"
        )
    return metadata_path
