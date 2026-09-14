import json
import csv
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from video_bench.config import load_config
from video_bench.generators import GenerationResult
from video_bench.prompts import PromptRecord
from video_bench.runner import run_generation


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value[0]


class _FakeSplit:
    def __init__(self, jobs):
        self.jobs = jobs

    def __enter__(self):
        return self.jobs

    def __exit__(self, *args):
        return None


class _FakeAccelerator:
    is_main_process = True
    num_processes = 1
    process_index = 0
    device = "cpu"

    def wait_for_everyone(self):
        return None

    def split_between_processes(self, jobs, apply_padding=False):
        assert apply_padding is False
        return _FakeSplit(jobs)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        return tensor


@pytest.fixture(autouse=True)
def fake_runtime_modules(monkeypatch):
    accelerate = types.ModuleType("accelerate")
    accelerate.Accelerator = _FakeAccelerator
    torch = types.ModuleType("torch")
    torch.tensor = lambda value, device=None: _FakeTensor(value)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "torch", torch)


class _StubGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, seed, output_path, **kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        self.calls.append((prompt, seed, path, kwargs))
        return GenerationResult(path, kwargs, kwargs["num_frames"])


class _FailingGenerator:
    def generate(self, prompt, seed, output_path, **kwargs):
        raise OSError("encoder failed")


def _config():
    config = load_config("configs/wan.yaml")
    return replace(config, runtime=replace(config.runtime, offload=None))


def test_run_writes_metadata_and_resumes_only_complete_jobs(tmp_path):
    prompt = PromptRecord("clip", "a test prompt")
    generator = _StubGenerator()
    factory = lambda *_: generator

    metadata_path = run_generation(
        _config(), [prompt], [7], output_dir=tmp_path, generator_factory=factory
    )
    assert len(generator.calls) == 1
    record = json.loads(metadata_path.read_text().strip())
    assert record["output_path"] == "videos/clip/seed-7.mp4"
    assert record["seed"] == 7

    run_generation(_config(), [prompt], [7], output_dir=tmp_path, generator_factory=factory)
    assert len(generator.calls) == 1

    metadata_path.unlink()
    run_generation(_config(), [prompt], [7], output_dir=tmp_path, generator_factory=factory)
    assert len(generator.calls) == 2


def test_csv_manifest_is_written_immediately_and_used_to_resume(tmp_path):
    prompt = PromptRecord("clip", "a test prompt")
    generator = _StubGenerator()
    manifest = tmp_path / "manifest.csv"

    result = run_generation(
        _config(), [prompt], [7], output_dir=tmp_path,
        manifest_path=manifest, videos_dir=tmp_path / "videos",
        generator_factory=lambda *_: generator,
    )

    assert result == manifest
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["model"] == "wan-2.1-t2v-1.3b"
    assert rows[0]["output_path"] == "videos/wan-2.1-t2v-1.3b/clip/seed-7.mp4"

    run_generation(
        _config(), [prompt], [7], output_dir=tmp_path,
        manifest_path=manifest, videos_dir=tmp_path / "videos",
        generator_factory=lambda *_: generator,
    )
    assert len(generator.calls) == 1


def test_dry_run_validates_without_creating_output(tmp_path):
    generator = _StubGenerator()
    metadata_path = run_generation(
        _config(), [PromptRecord("clip", "prompt")], [0], output_dir=tmp_path,
        dry_run=True, generator_factory=lambda *_: generator,
    )

    assert not metadata_path.parent.exists()
    assert generator.calls == []


@pytest.mark.parametrize("seeds", [[], [1, 1], [True]])
def test_invalid_seeds_are_rejected(tmp_path, seeds):
    with pytest.raises(ValueError, match="seed|Seed"):
        run_generation(
            _config(), [PromptRecord("clip", "prompt")], seeds, output_dir=tmp_path,
            dry_run=True, generator_factory=lambda *_: _StubGenerator(),
        )


def test_prompt_override_values_are_validated(tmp_path):
    prompt = PromptRecord("clip", "prompt", generation={"fps": 0})

    with pytest.raises(ValueError, match="generation.fps must be positive"):
        run_generation(
            _config(), [prompt], [0], output_dir=tmp_path, dry_run=True,
            generator_factory=lambda *_: _StubGenerator(),
        )


def test_failures_are_recorded_and_reported(tmp_path):
    with pytest.raises(RuntimeError, match="1 generation job"):
        run_generation(
            _config(), [PromptRecord("clip", "prompt")], [3], output_dir=tmp_path,
            generator_factory=lambda *_: _FailingGenerator(),
        )

    error_path = tmp_path / "wan-2.1-t2v-1.3b" / ".errors.rank-0.jsonl"
    error = json.loads(error_path.read_text().strip())
    assert error["prompt_id"] == "clip"
    assert error["seed"] == 3
    assert error["error"] == "OSError: encoder failed"
