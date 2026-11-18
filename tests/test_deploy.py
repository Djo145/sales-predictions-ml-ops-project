"""Tests du point d'entrée de déploiement Prefect, sans serveur Prefect."""

from datetime import timedelta
from unittest.mock import Mock

import pytest

from retail_forecast import deploy


def test_build_serve_kwargs_uses_absolute_config_and_one_local_run(tmp_path):
    config_path = tmp_path / "config.json"

    kwargs = deploy.build_serve_kwargs(config_path)

    assert kwargs["name"] == "retail-sales-training-local"
    assert kwargs["parameters"] == {"config_path": str(config_path.resolve())}
    assert kwargs["limit"] == 1
    assert "interval" not in kwargs
    assert kwargs["version"] == "0.1.0"


def test_build_serve_kwargs_adds_optional_schedule(tmp_path):
    kwargs = deploy.build_serve_kwargs(tmp_path / "config.json", interval_minutes=15)

    assert kwargs["interval"] == timedelta(minutes=15)


def test_main_calls_flow_serve_with_requested_parameters(monkeypatch, tmp_path):
    serve = Mock()
    monkeypatch.setattr(deploy.training_flow, "serve", serve)
    config_path = tmp_path / "config.json"

    deploy.main(
        [
            "--config",
            str(config_path),
            "--name",
            "retail-demo",
            "--interval-minutes",
            "10",
        ]
    )

    serve.assert_called_once()
    kwargs = serve.call_args.kwargs
    assert kwargs["name"] == "retail-demo"
    assert kwargs["parameters"] == {"config_path": str(config_path.resolve())}
    assert kwargs["interval"] == timedelta(minutes=10)
    assert kwargs["limit"] == 1


def test_positive_minutes_rejects_zero_and_negative():
    with pytest.raises(Exception, match="strictement positif"):
        deploy.positive_minutes("0")
    with pytest.raises(Exception, match="strictement positif"):
        deploy.positive_minutes("-1")
