import json

import pytest

from video_bench.config import ConfigError, load_config, load_suite_config
from video_bench.prompts import load_prompts


def test_sample_configs_load():
    for path in (
        "configs/wan.yaml",
        "configs/hunyuan_video.yaml",
        "configs/cogvideox.yaml",
        "configs/ltx_video.yaml",
        "configs/ltx2.yaml",
    ):
        config = load_config(path)
        assert config.model.checkpoint
        assert config.generation.num_frames > 0


def test_prompt_formats_and_overrides(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {"prompts": [
                "a plain prompt",
                {"id": "custom id", "prompt": "another", "generation": {"fps": 12}},
            ]}
        )
    )
    prompts = load_prompts(path)
    assert prompts[0].id == "prompt-00000"
    assert prompts[1].id == "custom-id"
    assert prompts[1].generation == {"fps": 12}


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  name: x\n  adapter: wan\n  checkpoint: x/y\n  typo: true\n"
    )
    with pytest.raises(ConfigError, match="Unknown model key"):
        load_config(path)


def test_default_suite_selects_models_prompts_seeds_and_outputs():
    suite = load_suite_config("config.yaml")

    assert [path.name for path in suite.models] == ["cogvideox.yaml"]
    assert suite.prompts.path.name == "prompts.json"
    assert suite.prompts.count is None
    assert suite.seeds.values() == [0]
    assert suite.output.root_dir.name == "generated-videos"
    assert suite.output.videos_path == suite.output.root_dir / "videos"
    assert suite.output.manifest_path == suite.output.root_dir / "manifest.csv"
    assert suite.runtime.load_to_vram is True
    assert suite.runtime.local_files_only is True
    assert suite.runtime.model_cache_dir is not None
    assert not suite.runtime.model_cache_dir.is_relative_to(suite.source_path.parent)
