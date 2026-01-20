import os
from pathlib import Path

from ugro.result_aggregator import ResultAggregator


def test_result_aggregator_creates_expected_layout(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UGRO_DATA_DIR", str(tmp_path / "ugro_data"))

    aggregator = ResultAggregator()
    paths = aggregator.ensure_job_layout("job_001")

    assert paths.job_dir == tmp_path / "ugro_data" / "jobs" / "job_001"
    assert paths.config_json == paths.job_dir / "config.json"
    assert paths.metrics_jsonl == paths.job_dir / "metrics.jsonl"
    assert paths.logs_dir == paths.job_dir / "logs"
    assert paths.checkpoints_dir == paths.job_dir / "checkpoints"
    assert paths.tensorboard_dir == paths.job_dir / "tensorboard"

    assert paths.logs_dir.is_dir()
    assert paths.checkpoints_dir.is_dir()
    assert paths.tensorboard_dir.is_dir()
    assert paths.metrics_jsonl.exists()


def test_result_aggregator_writes_config_and_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UGRO_DATA_DIR", str(tmp_path / "ugro_data"))

    aggregator = ResultAggregator()

    config_path = aggregator.write_job_config(
        "job_001",
        {"job_id": "job_001", "model": "tiny", "dataset": "wikitext"},
    )
    assert config_path.exists()

    metrics_path = aggregator.append_metrics(
        "job_001",
        {"timestamp": "2026-01-20T12:05:30Z", "job_id": "job_001", "rank": 0, "training_loss": 1.23},
    )
    assert metrics_path.exists()

    text = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 1
    assert '"job_id": "job_001"' in text[0]


def test_rank_log_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UGRO_DATA_DIR", str(tmp_path / "ugro_data"))

    aggregator = ResultAggregator()
    log_path = aggregator.rank_log_path("job_001", 2)

    assert log_path == tmp_path / "ugro_data" / "jobs" / "job_001" / "logs" / "rank_2.log"
