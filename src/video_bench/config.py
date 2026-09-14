"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a benchmark configuration is invalid."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    return dict(value)


def _reject_unknown(data: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {location} key(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    adapter: str
    checkpoint: str
    pipeline_class: str | None = None
    dtype: str = "bfloat16"
    revision: str | None = None
    variant: str | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    components: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> ModelConfig:
        data = _mapping(raw, "model")
        _reject_unknown(
            data,
            {
                "name", "adapter", "checkpoint", "pipeline_class", "dtype",
                "revision", "variant", "load_kwargs", "components",
            },
            "model",
        )
        for required in ("name", "adapter", "checkpoint"):
            if not isinstance(data.get(required), str) or not data[required].strip():
                raise ConfigError(f"model.{required} must be a non-empty string")
        components = _mapping(data.get("components"), "model.components")
        for name, spec in components.items():
            if not isinstance(name, str) or not isinstance(spec, dict):
                raise ConfigError("model.components must map component names to mappings")
        return cls(
            name=data["name"], adapter=data["adapter"], checkpoint=data["checkpoint"],
            pipeline_class=data.get("pipeline_class"), dtype=data.get("dtype", "bfloat16"),
            revision=data.get("revision"), variant=data.get("variant"),
            load_kwargs=_mapping(data.get("load_kwargs"), "model.load_kwargs"),
            components=components,
        )


@dataclass(frozen=True)
class GenerationConfig:
    width: int
    height: int
    num_frames: int
    fps: float
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> GenerationConfig:
        data = _mapping(raw, "generation")
        _reject_unknown(
            data,
            {"width", "height", "num_frames", "fps", "num_inference_steps",
             "guidance_scale", "negative_prompt", "parameters"},
            "generation",
        )
        defaults = {
            "width": 768, "height": 512, "num_frames": 81, "fps": 16.0,
            "num_inference_steps": 30, "guidance_scale": 5.0,
        }
        values = {key: data.get(key, value) for key, value in defaults.items()}
        for key in ("width", "height", "num_frames", "num_inference_steps"):
            if not isinstance(values[key], int) or isinstance(values[key], bool) or values[key] <= 0:
                raise ConfigError(f"generation.{key} must be a positive integer")
        for key in ("fps", "guidance_scale"):
            if not isinstance(values[key], (int, float)) or isinstance(values[key], bool):
                raise ConfigError(f"generation.{key} must be a number")
        if values["fps"] <= 0:
            raise ConfigError("generation.fps must be positive")
        negative_prompt = data.get("negative_prompt")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ConfigError("generation.negative_prompt must be a string or null")
        parameters = _mapping(data.get("parameters"), "generation.parameters")
        protected = {"prompt", "generator"} & set(parameters)
        if protected:
            raise ConfigError("generation.parameters cannot override prompt or generator")
        return cls(**values, negative_prompt=negative_prompt, parameters=parameters)

    def pipeline_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "width": self.width, "height": self.height, "num_frames": self.num_frames,
            "fps": self.fps, "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None:
            params["negative_prompt"] = self.negative_prompt
        params.update(self.parameters)
        return params


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str = "outputs"
    model_cache_dir: str | None = None
    local_files_only: bool = False
    offload: str | None = None
    vae_tiling: bool = True
    vae_slicing: bool = False
    attention_slicing: bool = False
    torch_compile: bool = False
    overwrite: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> RuntimeConfig:
        data = _mapping(raw, "runtime")
        _reject_unknown(
            data,
            {"output_dir", "model_cache_dir", "local_files_only", "offload", "vae_tiling",
             "vae_slicing", "attention_slicing", "torch_compile", "overwrite"},
            "runtime",
        )
        if data.get("offload") not in (None, "model", "sequential"):
            raise ConfigError("runtime.offload must be null, 'model', or 'sequential'")
        for key in (
            "local_files_only", "vae_tiling", "vae_slicing", "attention_slicing",
            "torch_compile", "overwrite",
        ):
            if key in data and not isinstance(data[key], bool):
                raise ConfigError(f"runtime.{key} must be a boolean")
        output_dir = data.get("output_dir", "outputs")
        if not isinstance(output_dir, str) or not output_dir:
            raise ConfigError("runtime.output_dir must be a non-empty string")
        model_cache_dir = data.get("model_cache_dir")
        if model_cache_dir is not None and (
            not isinstance(model_cache_dir, str) or not model_cache_dir
        ):
            raise ConfigError("runtime.model_cache_dir must be a non-empty string or null")
        return cls(**data)


@dataclass(frozen=True)
class BenchmarkConfig:
    model: ModelConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    source_path: Path


@dataclass(frozen=True)
class PromptSelectionConfig:
    path: Path
    count: int | None = None


@dataclass(frozen=True)
class SeedSelectionConfig:
    count: int
    start: int = 0

    def values(self) -> list[int]:
        return list(range(self.start, self.start + self.count))


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    videos_dir: str = "videos"
    manifest: str = "manifest.csv"

    @property
    def videos_path(self) -> Path:
        return self.root_dir / self.videos_dir

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / self.manifest


@dataclass(frozen=True)
class SuiteRuntimeConfig:
    model_cache_dir: Path | None = None
    load_to_vram: bool = True
    local_files_only: bool = False


@dataclass(frozen=True)
class SuiteConfig:
    """Top-level selection of models, prompts, seeds, and output locations."""

    models: tuple[Path, ...]
    prompts: PromptSelectionConfig
    seeds: SeedSelectionConfig
    output: OutputConfig
    runtime: SuiteRuntimeConfig
    source_path: Path


def _resolve_path(value: str, base_dir: Path, location: str) -> Path:
    import os

    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ConfigError(f"{location} contains an undefined environment variable: {value}")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_suite_config(path: str | Path) -> SuiteConfig:
    """Load the high-level configuration consumed by ``generate.py``."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    data = _mapping(raw, "configuration")
    _reject_unknown(data, {"models", "prompts", "seeds", "output", "runtime"}, "top-level")
    base_dir = config_path.parent

    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("models must be a non-empty list of model config paths")
    models: list[Path] = []
    for index, item in enumerate(raw_models):
        if isinstance(item, str):
            model_path, enabled = item, True
        elif isinstance(item, dict):
            _reject_unknown(item, {"config", "enabled"}, f"models[{index}]")
            model_path = item.get("config")
            enabled = item.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ConfigError(f"models[{index}].enabled must be a boolean")
        else:
            raise ConfigError(f"models[{index}] must be a path or mapping")
        if not isinstance(model_path, str) or not model_path:
            raise ConfigError(f"models[{index}].config must be a non-empty string")
        if enabled:
            models.append(_resolve_path(model_path, base_dir, f"models[{index}].config"))
    if not models:
        raise ConfigError("At least one model must be enabled")

    prompt_data = _mapping(data.get("prompts"), "prompts")
    _reject_unknown(prompt_data, {"path", "count"}, "prompts")
    prompt_path = prompt_data.get("path")
    if not isinstance(prompt_path, str) or not prompt_path:
        raise ConfigError("prompts.path must be a non-empty string")
    prompt_count = prompt_data.get("count")
    if prompt_count is not None and (
        not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count <= 0
    ):
        raise ConfigError("prompts.count must be a positive integer or null")

    seed_data = _mapping(data.get("seeds"), "seeds")
    _reject_unknown(seed_data, {"count", "start"}, "seeds")
    seed_count = seed_data.get("count")
    seed_start = seed_data.get("start", 0)
    if not isinstance(seed_count, int) or isinstance(seed_count, bool) or seed_count <= 0:
        raise ConfigError("seeds.count must be a positive integer")
    if not isinstance(seed_start, int) or isinstance(seed_start, bool):
        raise ConfigError("seeds.start must be an integer")

    output_data = _mapping(data.get("output"), "output")
    _reject_unknown(output_data, {"root_dir", "videos_dir", "manifest"}, "output")
    output_root = output_data.get("root_dir", "output")
    videos_dir = output_data.get("videos_dir", "videos")
    manifest = output_data.get("manifest", "manifest.csv")
    for key, value in {"root_dir": output_root, "videos_dir": videos_dir, "manifest": manifest}.items():
        if not isinstance(value, str) or not value:
            raise ConfigError(f"output.{key} must be a non-empty string")
    if Path(videos_dir).is_absolute() or Path(manifest).is_absolute():
        raise ConfigError("output.videos_dir and output.manifest must be relative to output.root_dir")
    resolved_output_root = _resolve_path(output_root, base_dir, "output.root_dir")
    if not (resolved_output_root / videos_dir).resolve().is_relative_to(resolved_output_root):
        raise ConfigError("output.videos_dir must stay inside output.root_dir")
    if not (resolved_output_root / manifest).resolve().is_relative_to(resolved_output_root):
        raise ConfigError("output.manifest must stay inside output.root_dir")

    runtime_data = _mapping(data.get("runtime"), "runtime")
    _reject_unknown(
        runtime_data, {"model_cache_dir", "load_to_vram", "local_files_only"}, "runtime",
    )
    cache_value = runtime_data.get("model_cache_dir")
    if cache_value is not None and (not isinstance(cache_value, str) or not cache_value):
        raise ConfigError("runtime.model_cache_dir must be a non-empty string or null")
    load_to_vram = runtime_data.get("load_to_vram", True)
    if not isinstance(load_to_vram, bool):
        raise ConfigError("runtime.load_to_vram must be a boolean")
    local_files_only = runtime_data.get("local_files_only", False)
    if not isinstance(local_files_only, bool):
        raise ConfigError("runtime.local_files_only must be a boolean")

    return SuiteConfig(
        models=tuple(models),
        prompts=PromptSelectionConfig(
            path=_resolve_path(prompt_path, base_dir, "prompts.path"), count=prompt_count,
        ),
        seeds=SeedSelectionConfig(count=seed_count, start=seed_start),
        output=OutputConfig(
            root_dir=resolved_output_root,
            videos_dir=videos_dir,
            manifest=manifest,
        ),
        runtime=SuiteRuntimeConfig(
            model_cache_dir=(
                _resolve_path(cache_value, base_dir, "runtime.model_cache_dir")
                if cache_value is not None else None
            ),
            load_to_vram=load_to_vram,
            local_files_only=local_files_only,
        ),
        source_path=config_path,
    )


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    data = _mapping(raw, "configuration")
    _reject_unknown(data, {"model", "generation", "runtime"}, "top-level")
    if "model" not in data:
        raise ConfigError("Missing required model section")
    return BenchmarkConfig(
        model=ModelConfig.from_dict(data["model"]),
        generation=GenerationConfig.from_dict(data.get("generation")),
        runtime=RuntimeConfig.from_dict(data.get("runtime")),
        source_path=config_path,
    )
