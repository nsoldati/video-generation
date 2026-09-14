from dataclasses import replace
import sys
import types

import pytest

from video_bench.config import load_config
from video_bench.generators import create_generator
from video_bench.generators.diffusers import LTX2Generator, LTXVideoGenerator
from video_bench.generators.diffusers import CogVideoXGenerator


def test_ltx_adapters_translate_export_fps_to_frame_rate():
    config = load_config("configs/ltx_video.yaml")
    generator = LTXVideoGenerator(config.model, config.runtime, "cpu")

    fps, params = generator.prepare_generation_parameters(
        {"fps": 12, "frame_rate": 99, "num_frames": 9}
    )

    assert fps == 12
    assert params == {"frame_rate": 12, "num_frames": 9}


def test_ltx2_adapter_requests_numpy_output():
    config = load_config("configs/ltx2.yaml")
    generator = LTX2Generator(config.model, config.runtime, "cpu")

    fps, params = generator.prepare_generation_parameters({"fps": 24})

    assert fps == 24
    assert params == {"frame_rate": 24, "output_type": "np"}


def test_generic_adapter_requires_pipeline_class_at_creation():
    config = load_config("configs/wan.yaml")
    model = replace(config.model, adapter="diffusers", pipeline_class=None)

    with pytest.raises(ValueError, match="requires model.pipeline_class"):
        create_generator(model, config.runtime, "cpu")


def test_external_model_cache_is_forced_for_pipeline_downloads(monkeypatch, tmp_path):
    calls = []

    class FakePipeline:
        vae = None

        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls.append((checkpoint, kwargs))
            return cls()

        def to(self, device):
            self.device = device

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "video_bench.generators.diffusers._resolve_class", lambda _: FakePipeline,
    )
    config = load_config("configs/cogvideox.yaml")
    runtime = replace(config.runtime, model_cache_dir=str(tmp_path), offload=None)

    CogVideoXGenerator(config.model, runtime, "cuda:0").load()

    assert calls[0][1]["cache_dir"] == str(tmp_path)
