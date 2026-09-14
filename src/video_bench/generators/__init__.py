"""Generator registry and built-in adapters."""

from __future__ import annotations

from typing import Any

from video_bench.config import ModelConfig, RuntimeConfig
from video_bench.generators.base import GenerationResult, VideoGenerator
from video_bench.generators.diffusers import (
    CogVideoXGenerator,
    DiffusersVideoGenerator,
    HunyuanVideoGenerator,
    LTX2Generator,
    LTXVideoGenerator,
    WanGenerator,
)

_GENERATORS: dict[str, type[VideoGenerator]] = {}


def register_generator(name: str, generator_type: type[VideoGenerator]) -> None:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Generator name cannot be empty")
    _GENERATORS[normalized] = generator_type


def create_generator(model: ModelConfig, runtime: RuntimeConfig, device: Any) -> VideoGenerator:
    adapter = model.adapter.strip().lower().replace("-", "_")
    try:
        generator_type = _GENERATORS[adapter]
    except KeyError as exc:
        available = ", ".join(sorted(_GENERATORS))
        raise ValueError(f"Unknown adapter {model.adapter!r}. Available adapters: {available}") from exc
    generator = generator_type(model, runtime, device)  # type: ignore[call-arg]
    if isinstance(generator, DiffusersVideoGenerator):
        # Resolve configuration errors without importing a pipeline or loading weights.
        generator.configured_pipeline_class
    return generator


for _name, _type in {
    "diffusers": DiffusersVideoGenerator,
    "wan": WanGenerator,
    "hunyuan_video": HunyuanVideoGenerator,
    "hunyuanvideo": HunyuanVideoGenerator,
    "cogvideox": CogVideoXGenerator,
    "ltx_video": LTXVideoGenerator,
    "ltx": LTXVideoGenerator,
    "ltx2": LTX2Generator,
    "ltx_2": LTX2Generator,
}.items():
    register_generator(_name, _type)

__all__ = ["GenerationResult", "VideoGenerator", "create_generator", "register_generator"]
