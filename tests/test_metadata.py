from video_bench.metadata import append_jsonl, merge_metadata, read_csv, read_jsonl, upsert_csv


def test_metadata_merge_deduplicates_new_records(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    shard = tmp_path / ".metadata.rank-0.jsonl"
    append_jsonl(metadata, {"output_path": "old.mp4", "prompt_id": "b", "seed": 1})
    append_jsonl(shard, {"output_path": "new.mp4", "prompt_id": "a", "seed": 2})
    append_jsonl(shard, {"output_path": "old.mp4", "prompt_id": "b", "seed": 1, "new": True})
    merge_metadata(metadata, [shard], overwrite=False)
    records = read_jsonl(metadata)
    assert [record["output_path"] for record in records] == ["new.mp4", "old.mp4"]
    assert records[1]["new"] is True
    assert not shard.exists()


def test_csv_upsert_replaces_rows_and_serializes_nested_values(tmp_path):
    manifest = tmp_path / "manifest.csv"
    upsert_csv(
        manifest,
        {"status": "completed", "model": "m", "prompt_id": "p", "seed": 1,
         "output_path": "videos/a.mp4", "prompt_metadata": {"split": "test"}},
    )
    upsert_csv(
        manifest,
        {"status": "completed", "model": "m", "prompt_id": "p", "seed": 1,
         "output_path": "videos/a.mp4", "generation_seconds": 2.5},
    )

    rows = read_csv(manifest)
    assert len(rows) == 1
    assert rows[0]["generation_seconds"] == "2.5"
