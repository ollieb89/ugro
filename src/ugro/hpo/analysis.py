"""HPO Results Analysis and Visualization.

Provides comprehensive analysis of HPO studies including parameter
importance, trial progression, and optimization visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def analyze_hpo_results(
    storage_backend: str,
    study_name: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive HPO results analysis.

    Loads an Optuna study and performs:
    - Best trial identification
    - Parameter importance analysis
    - Trial progression visualization
    - Parameter sensitivity plots
    - Objective distribution analysis

    Args:
        storage_backend: Optuna storage URI (e.g., "sqlite:///ugro_hpo.db")
        study_name: Name of the study to analyze
        output_dir: Directory for saving visualizations (default: current dir)

    Returns:
        Dictionary containing best_value, best_params, importance, trials_df
    """
    try:
        import matplotlib.pyplot as plt
        import optuna
        import pandas as pd
        from optuna.importance import get_param_importances
    except ImportError as e:
        raise ImportError(
            "Analysis requires optuna, pandas, and matplotlib. "
            "Install with: pip install optuna pandas matplotlib"
        ) from e

    # Load study from storage
    logger.info(f"Loading study '{study_name}' from {storage_backend}")
    storage = optuna.storages.RDBStorage(storage_backend)
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Convert to DataFrame
    trials_df = study.trials_dataframe()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"HPO Study: {study_name}")
    print(f"{'=' * 60}")
    print(f"Total Trials: {len(trials_df)}")

    completed = len(trials_df[trials_df["state"] == "COMPLETE"])
    print(f"Completed: {completed}")
    print(f"Best Value: {study.best_value:.6f}")

    print(f"\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Compute parameter importance
    try:
        importance = get_param_importances(study)
        print(f"\nParameter Importance:")
        for param, imp in sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute importance: {e}")
        importance = {}

    # Generate visualizations
    output_path = Path(output_dir) if output_dir else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)

    _create_visualizations(study, trials_df, importance, output_path, study_name)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "importance": importance,
        "trials_df": trials_df,
        "n_trials": len(trials_df),
        "n_completed": completed,
    }


def _create_visualizations(
    study: Any,
    trials_df: Any,
    importance: Dict[str, float],
    output_path: Path,
    study_name: str,
) -> None:
    """Generate HPO analysis visualizations.

    Args:
        study: Optuna study object
        trials_df: Trials DataFrame
        importance: Parameter importance dict
        output_path: Output directory
        study_name: Study name for file naming
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trial Progression
    ax = axes[0, 0]
    completed = trials_df[trials_df["state"] == "COMPLETE"]
    if not completed.empty and "value" in completed.columns:
        ax.plot(completed.index, completed["value"], "b-", alpha=0.7, linewidth=1)
        ax.axhline(
            y=study.best_value,
            color="r",
            linestyle="--",
            label=f"Best: {study.best_value:.4f}",
        )
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Objective Value")
        ax.set_title("Trial Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Parameter vs Performance (first parameter)
    ax = axes[0, 1]
    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    if param_cols and "value" in trials_df.columns:
        param_col = param_cols[0]
        param_name = param_col.replace("params_", "")

        valid = completed.dropna(subset=[param_col, "value"])
        if not valid.empty:
            ax.scatter(valid[param_col], valid["value"], alpha=0.6, c="blue")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Objective Value")
            ax.set_title(f"Parameter Sensitivity: {param_name}")
            ax.grid(True, alpha=0.3)

    # 3. Objective Distribution
    ax = axes[1, 0]
    if not completed.empty and "value" in completed.columns:
        valid_values = completed["value"].dropna()
        if not valid_values.empty:
            ax.hist(valid_values, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(
                study.best_value,
                color="r",
                linestyle="--",
                label=f"Best: {study.best_value:.4f}",
            )
            ax.set_xlabel("Objective Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Objective Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 4. Parameter Importance
    ax = axes[1, 1]
    if importance:
        import_df = pd.DataFrame(
            list(importance.items()), columns=["Parameter", "Importance"]
        ).sort_values("Importance", ascending=True)

        ax.barh(
            import_df["Parameter"],
            import_df["Importance"],
            color="teal",
            alpha=0.8,
        )
        ax.set_xlabel("Importance")
        ax.set_title("Parameter Importance")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f"hpo_analysis_{study_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Visualization saved: {fig_path}")


def export_best_config(
    storage_backend: str,
    study_name: str,
    output_path: str,
) -> Dict[str, Any]:
    """Export best hyperparameters to YAML file.

    Args:
        storage_backend: Optuna storage URI
        study_name: Study name
        output_path: Output YAML file path

    Returns:
        Best parameters dictionary
    """
    try:
        import optuna
        import yaml
    except ImportError as e:
        raise ImportError(
            "Export requires optuna and pyyaml. "
            "Install with: pip install optuna pyyaml"
        ) from e

    # Load study
    storage = optuna.storages.RDBStorage(storage_backend)
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Get best params
    best_params = study.best_params

    # Write to YAML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Best config exported to {output_file}")

    return best_params


def compare_studies(
    storage_backend: str,
    study_names: list[str],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare multiple HPO studies.

    Args:
        storage_backend: Optuna storage URI
        study_names: List of study names to compare
        output_path: Optional output file for comparison chart

    Returns:
        Comparison statistics dictionary
    """
    try:
        import matplotlib.pyplot as plt
        import optuna
    except ImportError as e:
        raise ImportError(
            "Comparison requires optuna and matplotlib."
        ) from e

    storage = optuna.storages.RDBStorage(storage_backend)
    results = []

    for name in study_names:
        try:
            study = optuna.load_study(study_name=name, storage=storage)
            results.append({
                "study": name,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "best_params": study.best_params,
            })
        except Exception as e:
            logger.warning(f"Could not load study '{name}': {e}")

    # Print comparison
    print(f"\n{'Study Comparison':^60}")
    print("=" * 60)
    for r in results:
        print(f"{r['study']}: best={r['best_value']:.6f}, trials={r['n_trials']}")

    # Create comparison chart if output specified
    if output_path and results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r["study"] for r in results]
        values = [r["best_value"] for r in results]

        ax.bar(names, values, color="steelblue", alpha=0.8)
        ax.set_xlabel("Study")
        ax.set_ylabel("Best Objective Value")
        ax.set_title("HPO Study Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Comparison chart saved: {output_path}")

    return {"studies": results}
