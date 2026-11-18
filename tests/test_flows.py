"""Tests des tâches Prefect, du cache et de la politique de retry."""

import errno
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prefect.cache_policies import NO_CACHE

from retail_forecast import flows, prefect_setup


def write_config(path, current_file="data/online_retail_actuel.xlsx"):
    config = {
        "data": {"current_file": current_file},
        "preprocessing": {
            "top_n": 10,
            "selection_end": "2011-08-31",
            "lookback": 100,
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


class FakeState:
    def __init__(self, error, failed=True):
        self.error = error
        self.failed = failed

    def is_failed(self):
        return self.failed

    def result(self, raise_on_failure=False):
        return self.error


def test_retry_policy_only_accepts_transient_io_errors():
    assert flows.retry_transient_ingestion(
        None, None, FakeState(TimeoutError("temporary timeout"))
    )
    assert flows.retry_transient_ingestion(
        None, None, FakeState(OSError(errno.EIO, "temporary read error"))
    )
    assert not flows.retry_transient_ingestion(
        None, None, FakeState(FileNotFoundError("missing file"))
    )
    assert not flows.retry_transient_ingestion(
        None, None, FakeState(ValueError("invalid data"))
    )
    assert not flows.retry_transient_ingestion(
        None, None, FakeState(TimeoutError("not failed"), failed=False)
    )


def test_preparation_cache_key_tracks_source_parameters_and_code_version():
    parameters = {
        "ingested": {"source_sha256": "source-a", "raw_data": object()},
        "config": {
            "preprocessing": {
                "top_n": 10,
                "selection_end": "2011-08-31",
                "lookback": 100,
            }
        },
    }
    original = flows.preparation_cache_key(None, parameters)
    assert flows.preparation_cache_key(None, parameters) == original
    assert flows.preparation_cache_key(
        SimpleNamespace(flow_run_id="different-flow"), parameters
    ) == original

    changed_source = {
        **parameters,
        "ingested": {"source_sha256": "source-b", "raw_data": object()},
    }
    assert flows.preparation_cache_key(None, changed_source) != original

    changed_config = json.loads(json.dumps(parameters["config"]))
    changed_config["preprocessing"]["top_n"] = 9
    assert flows.preparation_cache_key(
        None, {**parameters, "config": changed_config}
    ) != original

    old_version = flows.PREPARATION_VERSION
    flows.PREPARATION_VERSION = "daily-sales-test-v2"
    try:
        assert flows.preparation_cache_key(None, parameters) != original
    finally:
        flows.PREPARATION_VERSION = old_version


def test_ingestion_task_logs_counts_and_source_hash(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(flows, "get_run_logger", lambda: logger)
    monkeypatch.setattr(
        flows,
        "ingest_data",
        lambda path: {
            "raw_data": [1, 2, 3],
            "source_sha256": "abc123",
            "source_path": path,
        },
    )

    result = flows.ingestion_task.fn("/tmp/sales.xlsx")

    assert result["source_sha256"] == "abc123"
    assert any(
        "%s lignes" in call.args[0] and call.args[1] == 3
        for call in logger.info.call_args_list
    )


def test_flow_chains_five_tasks_and_tracks_flow_id(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config = write_config(config_path)
    logger = Mock()
    events = []
    run = SimpleNamespace(info=SimpleNamespace(run_id="mlflow-123"))
    ingested = {"raw_data": "raw", "source_sha256": "hash"}
    prepared = {"daily_sales": "daily"}
    training = {"models": "models"}
    evaluation = {"overall": "metrics"}
    publication = {"artifact_paths": {}, "metadata_paths": {}, "registered_models": {}}

    monkeypatch.setattr(flows, "get_run_logger", lambda: logger)
    monkeypatch.setattr(
        flows,
        "ensure_cache_storage_block",
        lambda: events.append(("cache-check",)) or (False, None),
    )
    monkeypatch.setattr(flows.flow_run, "id", "prefect-123")
    monkeypatch.setattr(
        flows,
        "start_tracking_run",
        lambda received_config, **kwargs: events.append(
            ("tracking-start", received_config, kwargs)
        ) or run,
    )
    monkeypatch.setattr(
        flows,
        "ingestion_task",
        lambda path: events.append(("ingestion", path)) or ingested,
    )
    monkeypatch.setattr(
        flows,
        "preparation_task",
        lambda source, received_config: events.append(("preparation", source, received_config))
        or prepared,
    )
    monkeypatch.setattr(
        flows,
        "training_task",
        lambda data, received_config: events.append(("training", data, received_config))
        or training,
    )
    monkeypatch.setattr(
        flows,
        "evaluation_task",
        lambda data, fit: events.append(("evaluation", data, fit)) or evaluation,
    )
    monkeypatch.setattr(
        flows,
        "publication_task",
        lambda **kwargs: events.append(("publication", kwargs)) or publication,
    )
    monkeypatch.setattr(
        flows,
        "finish_tracking_run",
        lambda run_id: events.append(("tracking-finish", run_id)),
    )
    monkeypatch.setattr(
        flows,
        "build_pipeline_result",
        lambda *args: {"assembled": args},
    )

    result = flows.training_flow.fn(
        config_path=str(config_path), output_dir="runs/manual-test"
    )

    assert result["assembled"] == (
        prepared,
        training,
        evaluation,
        publication,
        "mlflow-123",
    )
    assert [event[0] for event in events] == [
        "cache-check",
        "tracking-start",
        "ingestion",
        "preparation",
        "training",
        "evaluation",
        "publication",
        "tracking-finish",
    ]
    assert events[1][2] == {"flow_run_id": "prefect-123"}
    assert events[2][1] == str((tmp_path / "data/online_retail_actuel.xlsx").resolve())
    assert events[6][1]["mlflow_run_id"] == "mlflow-123"


def test_flow_marks_mlflow_failed_and_rethrows_original_error(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    write_config(config_path)
    run = SimpleNamespace(info=SimpleNamespace(run_id="mlflow-failed"))
    failure = ValueError("bad config")
    failures = []
    monkeypatch.setattr(flows, "get_run_logger", lambda: Mock())
    monkeypatch.setattr(flows, "ensure_cache_storage_block", lambda: (False, None))
    monkeypatch.setattr(flows, "start_tracking_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(flows, "ingestion_task", Mock(side_effect=failure))
    training_spy = Mock()
    publication_spy = Mock()
    monkeypatch.setattr(flows, "training_task", training_spy)
    monkeypatch.setattr(flows, "publication_task", publication_spy)
    monkeypatch.setattr(
        flows,
        "fail_tracking_run",
        lambda run_id, error: failures.append((run_id, error)),
    )

    with pytest.raises(ValueError, match="bad config"):
        flows.training_flow.fn(config_path=str(config_path), output_dir="runs/failure")

    assert failures == [("mlflow-failed", failure)]
    training_spy.assert_not_called()
    publication_spy.assert_not_called()


def test_prefect_task_settings_are_explicit():
    assert flows.training_flow.retries == 0
    assert flows.ingestion_task.retries == 2
    assert flows.ingestion_task.retry_delay_seconds == [2, 5]
    assert flows.ingestion_task.timeout_seconds == 300
    assert flows.ingestion_task.cache_policy == NO_CACHE
    assert flows.preparation_task.retries == 0
    assert flows.preparation_task.persist_result is True
    assert flows.preparation_task.cache_key_fn is flows.preparation_cache_key
    assert flows.preparation_task.result_storage == flows.PREFECT_RESULT_STORAGE_BLOCK
    assert flows.training_task.retries == 0
    assert flows.training_task.cache_policy == NO_CACHE
    assert flows.evaluation_task.cache_policy == NO_CACHE
    assert flows.publication_task.cache_policy == NO_CACHE


def test_cache_storage_setup_saves_local_filesystem_block(monkeypatch, tmp_path):
    cache_path = tmp_path / "prefect-results"
    calls = []

    class FakeStorage:
        def __init__(self, *, basepath):
            calls.append(("init", basepath))

        def save(self, *, name, overwrite):
            calls.append(("save", name, overwrite))
            return "block-id"

    monkeypatch.setattr(prefect_setup, "PREFECT_CACHE_DIR", cache_path)
    monkeypatch.setattr(prefect_setup, "LocalFileSystem", FakeStorage)

    assert prefect_setup.configure_cache_storage() == "block-id"
    assert cache_path.is_dir()
    assert calls == [
        ("init", str(cache_path)),
        ("save", "retail-forecast-cache", True),
    ]


def test_ensure_cache_block_creates_it_when_missing(monkeypatch):
    calls = []

    class MissingStorage:
        @classmethod
        def load(cls, name):
            raise ValueError(
                f"Unable to find block document named {name} for block type local-file-system"
            )

    monkeypatch.setattr(prefect_setup, "LocalFileSystem", MissingStorage)
    monkeypatch.setattr(
        prefect_setup,
        "configure_cache_storage",
        lambda: calls.append("create") or "created-id",
    )

    assert prefect_setup.ensure_cache_storage_block() == (True, "created-id")
    assert calls == ["create"]


def test_ensure_cache_block_keeps_existing_block(monkeypatch):
    class ExistingStorage:
        @classmethod
        def load(cls, name):
            return object()

    monkeypatch.setattr(prefect_setup, "LocalFileSystem", ExistingStorage)
    monkeypatch.setattr(
        prefect_setup,
        "configure_cache_storage",
        Mock(side_effect=AssertionError("Ne doit pas recréer le block existant")),
    )

    assert prefect_setup.ensure_cache_storage_block() == (False, None)
