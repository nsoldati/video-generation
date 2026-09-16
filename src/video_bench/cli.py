"""Command-line interface."""

from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from video_bench.config import ConfigError, load_config, load_suite_config
from video_bench.prompts import load_prompts
from video_bench.runner import run_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible text-to-video benchmark dataset."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="High-level YAML configuration (default: config.yaml)",
    )
    parser.add_argument("--prompts", help="Override the configured JSON prompt dataset")
    parser.add_argument(
        "--prompt-count", type=int,
        help="Override prompts.count (useful for small validation runs)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the configured seeds")
    parser.add_argument("--output-dir", help="Override output.root_dir")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing MP4 files")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list work without loading a model")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def _release_cuda_memory() -> None:
    """Release one model before the next selected model is loaded."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds contains duplicate values")
    try:
        suite = load_suite_config(args.config)
        prompt_path = Path(args.prompts).expanduser().resolve() if args.prompts else suite.prompts.path
        prompts = load_prompts(prompt_path)
        prompt_count = args.prompt_count if args.prompt_count is not None else suite.prompts.count
        if prompt_count is not None:
            if prompt_count <= 0:
                raise ConfigError("--prompt-count must be a positive integer")
            if prompt_count > len(prompts):
                raise ConfigError(
                    f"prompt count is {prompt_count}, but {prompt_path} contains "
                    f"only {len(prompts)} prompts"
                )
            prompts = prompts[:prompt_count]
        seeds = args.seeds if args.seeds is not None else suite.seeds.values()
        output_root = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir else suite.output.root_dir
        )
        videos_path = output_root / suite.output.videos_dir
        manifest_path = output_root / suite.output.manifest

        selected_models = []
        for model_path in suite.models:
            model_config = load_config(model_path)
            runtime = replace(
                model_config.runtime,
                output_dir=str(output_root),
                model_cache_dir=(
                    str(suite.runtime.model_cache_dir)
                    if suite.runtime.model_cache_dir is not None
                    else model_config.runtime.model_cache_dir
                ),
                local_files_only=suite.runtime.local_files_only,
                offload=None if suite.runtime.load_to_vram else model_config.runtime.offload,
            )
            selected_models.append((model_path, replace(model_config, runtime=runtime)))
        model_names = [config.model.name for _, config in selected_models]
        if len(set(model_names)) != len(model_names):
            raise ConfigError("Enabled models must have unique model.name values")

        for model_path, model_config in selected_models:
            logging.info("Selected model %s (%s)", model_config.model.name, model_path)
            try:
                run_generation(
                    model_config, prompts, seeds, output_dir=output_root,
                    manifest_path=manifest_path, videos_dir=videos_path,
                    overwrite=True if args.overwrite else None, dry_run=args.dry_run,
                )
            finally:
                _release_cuda_memory()
    except (ConfigError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.dry_run:
        logging.info("Dry run completed successfully")
    else:
        logging.info("Manifest written to %s", manifest_path)
