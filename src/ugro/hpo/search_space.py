"""HPO Search Space Parser.

Handles loading and parsing YAML search space configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ugro.hpo.config import Objective, ParameterBound

logger = logging.getLogger(__name__)


def load_search_space_yaml(path: str | Path) -> Dict[str, Any]:
    """Load HPO search space configuration from YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Parsed YAML configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Search space config not found: {path}")

    logger.info(f"Loading search space from: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def parse_parameter_bounds(config: Dict[str, Any]) -> List[ParameterBound]:
    """Parse YAML parameters into ParameterBound objects.

    Args:
        config: Parsed YAML config with 'parameters' key

    Returns:
        List of ParameterBound objects

    Example YAML structure:
        parameters:
          learning_rate:
            type: float
            min: 1e-5
            max: 1e-2
            log: true
            default: 5e-4
    """
    bounds: List[ParameterBound] = []

    parameters = config.get("parameters", {})
    if not parameters:
        logger.warning("No parameters found in search space config")
        return bounds

    for param_name, spec in parameters.items():
        if not isinstance(spec, dict):
            logger.warning(f"Skipping invalid parameter spec: {param_name}")
            continue

        param_type = spec.get("type")
        if param_type not in ("int", "float", "categorical"):
            raise ValueError(
                f"Invalid parameter type '{param_type}' for {param_name}. "
                "Must be 'int', 'float', or 'categorical'."
            )

        bound = ParameterBound(
            name=param_name,
            type=param_type,
            min=spec.get("min"),
            max=spec.get("max"),
            choices=spec.get("choices"),
            log=spec.get("log", False),
            step=spec.get("step"),
            default=spec.get("default"),
        )
        bounds.append(bound)
        logger.debug(f"Parsed parameter: {bound}")

    logger.info(f"Parsed {len(bounds)} parameter bounds")
    return bounds


def parse_objectives(config: Dict[str, Any]) -> List[Objective]:
    """Parse YAML objectives into Objective objects.

    Args:
        config: Parsed YAML config with 'objectives' key

    Returns:
        List of Objective objects (defaults to minimize eval_loss)

    Example YAML structure:
        objectives:
          - name: accuracy
            direction: maximize
            weight: 0.7
          - name: latency_ms
            direction: minimize
            weight: 0.3
    """
    objectives: List[Objective] = []

    obj_list = config.get("objectives", [])
    if not obj_list:
        # Default objective
        logger.info("No objectives specified, defaulting to minimize eval_loss")
        return [Objective("eval_loss", "minimize")]

    for obj_spec in obj_list:
        if not isinstance(obj_spec, dict):
            logger.warning(f"Skipping invalid objective spec: {obj_spec}")
            continue

        name = obj_spec.get("name")
        direction = obj_spec.get("direction", "minimize")
        weight = float(obj_spec.get("weight", 1.0))

        if not name:
            raise ValueError("Objective must have a 'name' field")

        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"Invalid direction '{direction}' for objective {name}. "
                "Must be 'maximize' or 'minimize'."
            )

        objectives.append(Objective(name=name, direction=direction, weight=weight))
        logger.debug(f"Parsed objective: {name} ({direction}, weight={weight})")

    logger.info(f"Parsed {len(objectives)} objectives")
    return objectives


def parse_constraints(config: Dict[str, Any]) -> List[str]:
    """Parse constraint expressions from config.

    Args:
        config: Parsed YAML config with 'constraints' key

    Returns:
        List of constraint expression strings

    Example YAML structure:
        constraints:
          - "lora_alpha >= lora_r"
          - "batch_size <= 128"
    """
    return config.get("constraints", [])


def build_ray_search_space(
    bounds: List[ParameterBound],
) -> Dict[str, Any]:
    """Convert ParameterBounds to Ray Tune search space format.

    Args:
        bounds: List of parameter bounds

    Returns:
        Dictionary compatible with Ray Tune's param_space

    Note:
        Requires ray[tune] to be installed. Import is deferred.
    """
    try:
        from ray import tune
    except ImportError as e:
        raise ImportError(
            "Ray Tune is required for HPO. Install with: pip install 'ray[tune]'"
        ) from e

    ray_space: Dict[str, Any] = {}

    for param in bounds:
        if param.type == "int":
            if param.log:
                ray_space[param.name] = tune.lograndint(
                    int(param.min), int(param.max)
                )
            elif param.step:
                ray_space[param.name] = tune.qrandint(
                    int(param.min), int(param.max), int(param.step)
                )
            else:
                ray_space[param.name] = tune.randint(int(param.min), int(param.max))

        elif param.type == "float":
            if param.log:
                ray_space[param.name] = tune.loguniform(param.min, param.max)
            else:
                ray_space[param.name] = tune.uniform(param.min, param.max)

        elif param.type == "categorical":
            ray_space[param.name] = tune.choice(param.choices)

    return ray_space


def apply_parameter_to_optuna_trial(
    trial: Any,  # optuna.Trial
    param: ParameterBound,
) -> Any:
    """Apply a parameter bound to an Optuna trial.

    Args:
        trial: Optuna Trial object
        param: Parameter bound specification

    Returns:
        Sampled parameter value
    """
    if param.type == "int":
        return trial.suggest_int(
            param.name,
            int(param.min),
            int(param.max),
            step=int(param.step) if param.step else 1,
            log=param.log,
        )
    elif param.type == "float":
        return trial.suggest_float(
            param.name,
            param.min,
            param.max,
            log=param.log,
        )
    elif param.type == "categorical":
        return trial.suggest_categorical(param.name, param.choices)
    else:
        raise ValueError(f"Unknown parameter type: {param.type}")


def sample_defaults(bounds: List[ParameterBound]) -> Dict[str, Any]:
    """Extract default values from parameter bounds.

    Args:
        bounds: List of parameter bounds

    Returns:
        Dictionary of parameter name -> default value
    """
    defaults: Dict[str, Any] = {}
    for param in bounds:
        if param.default is not None:
            defaults[param.name] = param.default
    return defaults
