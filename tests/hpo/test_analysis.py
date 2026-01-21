"""Tests for HPO analysis and visualization module.

Note: Tests that require matplotlib visualization are marked with
@pytest.mark.integration as they need the real optuna/matplotlib stack.
The unit tests focus on the export and comparison logic where mocking
is simpler.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExportBestConfig:
    """Tests for export_best_config function."""

    def test_export_creates_yaml(self) -> None:
        """Test that export_best_config creates a YAML file."""
        # Setup mock
        mock_study = MagicMock()
        mock_study.best_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
        }

        mock_optuna = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = MagicMock()
        mock_optuna.load_study.return_value = mock_study

        with patch.dict(sys.modules, {"optuna": mock_optuna}):
            from ugro.hpo.analysis import export_best_config

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "best_config.yaml"
                result = export_best_config(
                    storage_backend="sqlite:///test.db",
                    study_name="test-study",
                    output_path=str(output_path),
                )

                assert output_path.exists()
                assert result == mock_study.best_params

                # Verify YAML content
                import yaml

                with open(output_path) as f:
                    loaded = yaml.safe_load(f)
                assert loaded["learning_rate"] == 0.001
                assert loaded["batch_size"] == 32

    def test_export_creates_parent_dirs(self) -> None:
        """Test that export creates parent directories if needed."""
        mock_study = MagicMock()
        mock_study.best_params = {"lr": 0.01}

        mock_optuna = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = MagicMock()
        mock_optuna.load_study.return_value = mock_study

        with patch.dict(sys.modules, {"optuna": mock_optuna}):
            from ugro.hpo.analysis import export_best_config

            with tempfile.TemporaryDirectory() as tmpdir:
                nested_path = Path(tmpdir) / "nested" / "dir" / "config.yaml"
                export_best_config(
                    storage_backend="sqlite:///test.db",
                    study_name="test",
                    output_path=str(nested_path),
                )

                assert nested_path.exists()


class TestCompareStudiesDataOnly:
    """Tests for compare_studies function - data path only.

    Note: compare_studies imports matplotlib internally for optional charting.
    These tests focus on the data comparison path (no output_path provided).
    """

    @pytest.fixture(autouse=True)
    def setup_optuna_mock(self):
        """Set up clean matplotlib mock for all tests."""
        # Mock matplotlib at the top level
        self.mock_plt = MagicMock()
        with patch.dict(sys.modules, {"matplotlib": MagicMock(), "matplotlib.pyplot": self.mock_plt}):
            yield

    def test_compare_returns_results_list(self) -> None:
        """Test that compare_studies returns comparison data."""
        # Setup mock studies
        study1 = MagicMock()
        study1.best_value = 0.1
        study1.trials = [MagicMock()] * 3
        study1.best_params = {"lr": 0.001}

        study2 = MagicMock()
        study2.best_value = 0.2
        study2.trials = [MagicMock()] * 5
        study2.best_params = {"lr": 0.01}

        mock_optuna = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = MagicMock()
        mock_optuna.load_study.side_effect = [study1, study2]

        with patch.dict(sys.modules, {"optuna": mock_optuna}):
            from ugro.hpo.analysis import compare_studies

            result = compare_studies(
                storage_backend="sqlite:///test.db",
                study_names=["study1", "study2"],
            )

        assert "studies" in result
        assert len(result["studies"]) == 2
        assert result["studies"][0]["study"] == "study1"
        assert result["studies"][0]["best_value"] == 0.1
        assert result["studies"][1]["study"] == "study2"
        assert result["studies"][1]["n_trials"] == 5

    def test_compare_handles_missing_study(self) -> None:
        """Test graceful handling when a study cannot be loaded."""
        study1 = MagicMock()
        study1.best_value = 0.1
        study1.trials = []
        study1.best_params = {}

        mock_optuna = MagicMock()
        mock_optuna.storages.RDBStorage.return_value = MagicMock()
        mock_optuna.load_study.side_effect = [
            study1,
            KeyError("Study not found"),
        ]

        with patch.dict(sys.modules, {"optuna": mock_optuna}):
            from ugro.hpo.analysis import compare_studies

            result = compare_studies(
                storage_backend="sqlite:///test.db",
                study_names=["exists", "missing"],
            )

        # Should only have one result (the successful one)
        assert len(result["studies"]) == 1
        assert result["studies"][0]["study"] == "exists"


class TestImportDependencies:
    """Tests for import error handling."""

    def test_analysis_module_importable(self) -> None:
        """Test that the analysis module can be imported."""
        from ugro.hpo import analysis

        assert hasattr(analysis, "analyze_hpo_results")
        assert hasattr(analysis, "export_best_config")
        assert hasattr(analysis, "compare_studies")

    def test_functions_are_callable(self) -> None:
        """Test that analysis functions are callable."""
        from ugro.hpo.analysis import (
            analyze_hpo_results,
            compare_studies,
            export_best_config,
        )

        assert callable(analyze_hpo_results)
        assert callable(export_best_config)
        assert callable(compare_studies)


class TestParameterValidation:
    """Tests for function signature and parameter validation."""

    def test_export_best_config_requires_all_params(self) -> None:
        """Test that export_best_config has expected signature."""
        from ugro.hpo.analysis import export_best_config
        import inspect

        sig = inspect.signature(export_best_config)
        params = list(sig.parameters.keys())

        assert "storage_backend" in params
        assert "study_name" in params
        assert "output_path" in params

    def test_compare_studies_optional_output(self) -> None:
        """Test that compare_studies has optional output_path."""
        from ugro.hpo.analysis import compare_studies
        import inspect

        sig = inspect.signature(compare_studies)
        output_param = sig.parameters.get("output_path")

        assert output_param is not None
        assert output_param.default is None  # Optional
