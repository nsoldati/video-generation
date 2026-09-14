"""Diffusers-backed model adapters."""

from __future__ import annotations

import importlib
import inspect
import os
import uuid
from pathlib import Path
from typing import Any

from video_bench.config import ModelConfig, RuntimeConfig
from video_bench.generators.base import GenerationResult, VideoGenerator


def _resolve_class(path: str) -> type[Any]:
    if "." not in path:
        module_name, class_name = "diffusers", path
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Could not find {class_name!r} in {module_name!r}") from exc


def _torch_dtype(name: str) -> Any:
    import torch

    aliases = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
    normalized = aliases.get(name.lower(), name.lower())
    dtype = getattr(torch, normalized, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


class DiffusersVideoGenerator(VideoGenerator):
    """Reusable adapter for Diffusers pipelines that return a ``frames`` field."""

    pipeline_class: str | None = None

    def __init__(self, model: ModelConfig, runtime: RuntimeConfig, device: Any) -> None:
        self.model = model
        self.runtime = runtime
        self.device = device
        self.pipeline: Any | None = None

    @property
    def configured_pipeline_class(self) -> str:
        pipeline_class = self.model.pipeline_class or self.pipeline_class
        if not pipeline_class:
            raise ValueError(
                "This adapter requires model.pipeline_class, for example "
                "'diffusers.MochiPipeline'"
            )
        return pipeline_class

    def _load_component(self, name: str, spec: dict[str, Any]) -> Any:
        component_spec = dict(spec)
        class_path = component_spec.pop("class", "diffusers.AutoModel")
        checkpoint = component_spec.pop("checkpoint", self.model.checkpoint)
        dtype_name = component_spec.pop("dtype", self.model.dtype)
        component_spec["torch_dtype"] = _torch_dtype(dtype_name)
        component_spec.setdefault("subfolder", name)
        if self.runtime.model_cache_dir is not None:
            component_spec["cache_dir"] = self.runtime.model_cache_dir
        if self.runtime.local_files_only:
            component_spec["local_files_only"] = True
        return _resolve_class(class_path).from_pretrained(checkpoint, **component_spec)

    def load(self) -> None:
        if self.pipeline is not None:
            return
        load_kwargs = dict(self.model.load_kwargs)
        load_kwargs["torch_dtype"] = _torch_dtype(self.model.dtype)
        if self.runtime.model_cache_dir is not None:
            load_kwargs["cache_dir"] = self.runtime.model_cache_dir
        if self.runtime.local_files_only:
            load_kwargs["local_files_only"] = True
        if self.model.revision is not None:
            load_kwargs["revision"] = self.model.revision
        if self.model.variant is not None:
            load_kwargs["variant"] = self.model.variant
        for name, spec in self.model.components.items():
            load_kwargs[name] = self._load_component(name, spec)

        pipeline_type = _resolve_class(self.configured_pipeline_class)
        self.pipeline = pipeline_type.from_pretrained(self.model.checkpoint, **load_kwargs)
        self._configure_pipeline()

    def _configure_pipeline(self) -> None:
        assert self.pipeline is not None
        vae = getattr(self.pipeline, "vae", None)
        if self.runtime.vae_tiling and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if self.runtime.vae_slicing and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if self.runtime.attention_slicing and hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        if self.runtime.offload:
            method_name = (
                "enable_model_cpu_offload"
                if self.runtime.offload == "model"
                else "enable_sequential_cpu_offload"
            )
            method = getattr(self.pipeline, method_name)
            signature = inspect.signature(method)
            offload_kwargs: dict[str, Any] = {}
            if "device" in signature.parameters:
                offload_kwargs["device"] = str(self.device)
            if "gpu_id" in signature.parameters and getattr(self.device, "index", None) is not None:
                offload_kwargs["gpu_id"] = self.device.index
            method(**offload_kwargs)
        else:
            self.pipeline.to(self.device)

        if self.runtime.torch_compile:
            import torch

            if not hasattr(self.pipeline, "transformer"):
                raise ValueError("torch_compile requires a pipeline.transformer component")
            self.pipeline.transformer = torch.compile(
                self.pipeline.transformer, mode="reduce-overhead", fullgraph=True
            )

    def prepare_generation_parameters(self, kwargs: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        params = dict(kwargs)
        fps = float(params.pop("fps"))
        return fps, params

    @staticmethod
    def _frames_from_output(output: Any) -> Any:
        frames = output.frames if hasattr(output, "frames") else output[0]
        return frames[0]

    def export_output(self, output: Any, output_path: Path, fps: float) -> int | None:
        from diffusers.utils import export_to_video

        frames = self._frames_from_output(output)
        export_to_video(frames, str(output_path), fps=fps)
        try:
            return len(frames)
        except TypeError:
            return None

    def generate(
        self,
        prompt: str,
        seed: int,
        output_path: str | Path,
        **kwargs: Any,
    ) -> GenerationResult:
        import torch

        self.load()
        assert self.pipeline is not None
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
        )
        fps, pipeline_parameters = self.prepare_generation_parameters(kwargs)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        try:
            output = self.pipeline(prompt=prompt, generator=generator, **pipeline_parameters)
            actual_frames = self.export_output(output, temp_path, fps)
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
        recorded_parameters = dict(pipeline_parameters)
        recorded_parameters["fps"] = fps
        return GenerationResult(destination, recorded_parameters, actual_frames)


class WanGenerator(DiffusersVideoGenerator):
    pipeline_class = "diffusers.WanPipeline"


class HunyuanVideoGenerator(DiffusersVideoGenerator):
    pipeline_class = "diffusers.HunyuanVideoPipeline"


class CogVideoXGenerator(DiffusersVideoGenerator):
    pipeline_class = "diffusers.CogVideoXPipeline"


class LTXVideoGenerator(DiffusersVideoGenerator):
    pipeline_class = "diffusers.LTXPipeline"

    def prepare_generation_parameters(self, kwargs: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        fps, params = super().prepare_generation_parameters(kwargs)
        params["frame_rate"] = fps
        return fps, params


class LTX2Generator(DiffusersVideoGenerator):
    pipeline_class = "diffusers.LTX2Pipeline"

    def prepare_generation_parameters(self, kwargs: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        fps, params = super().prepare_generation_parameters(kwargs)
        params["frame_rate"] = fps
        params.setdefault("output_type", "np")
        return fps, params

    def export_output(self, output: Any, output_path: Path, fps: float) -> int | None:
        audio = getattr(output, "audio", None)
        if audio is None and isinstance(output, tuple) and len(output) > 1:
            audio = output[1]
        if audio is None:
            return super().export_output(output, output_path, fps)
        from diffusers.utils import encode_video

        frames = self._frames_from_output(output)
        sample_rate = self.pipeline.vocoder.config.output_sampling_rate
        waveform = audio[0].float().cpu() if hasattr(audio[0], "float") else audio[0]
        encode_video(
            frames, fps=fps, audio=waveform, audio_sample_rate=sample_rate,
            output_path=str(output_path),
        )
        return len(frames)
