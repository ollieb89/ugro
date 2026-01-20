"""HPO Parameter Constraints.

Validates hyperparameter configurations against defined constraints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validate hyperparameters against constraint expressions.

    Constraints are Python expressions evaluated safely with only the
    parameter values in scope. They should evaluate to True for valid
    parameter combinations.

    Example:
        validator = ParameterValidator([
            "lora_alpha >= lora_r",
            "batch_size <= 128",
        ])
        valid = validator.validate({"lora_r": 16, "lora_alpha": 32, "batch_size": 64})
    """

    def __init__(self, constraints: List[str] | None = None):
        """Initialize validator with constraint expressions.

        Args:
            constraints: List of Python expressions that must evaluate to True
        """
        self.constraints = constraints or []

    def validate(self, params: Dict[str, Any]) -> bool:
        """Check if all constraints are satisfied.

        Args:
            params: Dictionary of parameter name -> value

        Returns:
            True if all constraints pass, False otherwise
        """
        for constraint in self.constraints:
            if not self._evaluate_constraint(constraint, params):
                return False
        return True

    def validate_with_errors(
        self, params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate and return list of violated constraints.

        Args:
            params: Dictionary of parameter name -> value

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        for constraint in self.constraints:
            if not self._evaluate_constraint(constraint, params):
                errors.append(f"Constraint violated: {constraint}")

        return len(errors) == 0, errors

    def _evaluate_constraint(
        self, constraint: str, params: Dict[str, Any]
    ) -> bool:
        """Safely evaluate a single constraint expression.

        Args:
            constraint: Python expression string
            params: Parameter values to use in evaluation

        Returns:
            True if constraint is satisfied, False otherwise
        """
        try:
            # Create a safe evaluation context with only parameters
            # No builtins to prevent code injection
            safe_globals: Dict[str, Any] = {"__builtins__": {}}
            safe_locals = dict(params)

            result = eval(constraint, safe_globals, safe_locals)

            if not result:
                logger.warning(
                    f"Constraint violation: {constraint} with params {params}"
                )
            return bool(result)

        except NameError as e:
            # Missing parameter in constraint - treat as invalid
            logger.error(f"Constraint evaluation error (missing param): {e}")
            return False

        except Exception as e:
            # Any other evaluation error
            logger.error(f"Constraint evaluation failed: {constraint} - {e}")
            return False


# Pre-defined constraints for common use cases

LORA_CONSTRAINTS: List[str] = [
    "lora_alpha >= lora_r",  # Alpha should be >= R
    "lora_dropout >= 0.0 and lora_dropout <= 0.5",  # Valid dropout range
    "learning_rate <= 0.01",  # Safety upper bound
    "batch_size <= 128",  # Memory constraint
]

TRANSFORMER_CONSTRAINTS: List[str] = [
    "learning_rate >= 1e-6",
    "learning_rate <= 1e-2",
    "warmup_ratio >= 0.0 and warmup_ratio <= 0.5",
    "weight_decay >= 0.0 and weight_decay <= 0.3",
]


def create_validator_from_config(
    constraints: List[str] | None = None,
    preset: str | None = None,
) -> ParameterValidator:
    """Create a validator from config constraints and/or preset.

    Args:
        constraints: Custom constraint expressions
        preset: Optional preset name ("lora", "transformer")

    Returns:
        Configured ParameterValidator instance
    """
    all_constraints: List[str] = []

    # Add preset constraints
    if preset == "lora":
        all_constraints.extend(LORA_CONSTRAINTS)
    elif preset == "transformer":
        all_constraints.extend(TRANSFORMER_CONSTRAINTS)

    # Add custom constraints
    if constraints:
        all_constraints.extend(constraints)

    return ParameterValidator(all_constraints)
