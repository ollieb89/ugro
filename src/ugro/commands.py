"""Command builder for UGRO operations."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass
class TrainingJobParams:
    """Parameters for a training job."""
    job_id: str
    model: str
    dataset: str
    epochs: int
    learning_rate: float
    nnodes: int
    nproc_per_node: int
    master_addr: str
    master_port: int
    node_rank: int
    script_path: str


class CommandBuilder:
    """Safely builds shell commands for remote execution."""

    @staticmethod
    def build_torchrun_command(params: TrainingJobParams) -> str:
        """Build a torchrun command with proper escaping.
        
        Args:
            params: Job parameters
            
        Returns:
            Shell-safe command string
        """
        # Base torchrun arguments
        cmd = [
            "torchrun",
            f"--nnodes={params.nnodes}",
            f"--nproc_per_node={params.nproc_per_node}",
            f"--rdzv_id={shlex.quote(params.job_id)}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={shlex.quote(f'{params.master_addr}:{params.master_port}')}",
            f"--node_rank={params.node_rank}",
            shlex.quote(params.script_path),
        ]

        # Script arguments
        script_args = [
            "--model", shlex.quote(params.model),
            "--dataset", shlex.quote(params.dataset),
            "--epochs", str(params.epochs),
            "--learning-rate", str(params.learning_rate),
            "--job-id", shlex.quote(params.job_id),
        ]
        
        return " ".join(cmd + script_args)

    @staticmethod
    def build_env_wrapper(command: str, env_type: str = "pixi", env_name: str = "cuda") -> str:
        """Wrap command in environment activation.
        
        Args:
            command: The command to wrap
            env_type: 'pixi' or 'conda'
            env_name: Name of environment
            
        Returns:
            Wrapped command string
        """
        if env_type == "pixi":
            # Check for pixi in common locations or PATH
            # We assume project root is current working directory where this runs
            if env_name == "default":
                return f"pixi run -- {command}"
            else:
                return f"pixi run -e {shlex.quote(env_name)} -- {command}"
        elif env_type == "conda":
            return f"conda run -n {shlex.quote(env_name)} -- {command}"
        else:
            # No environment wrapper - ensure command is available in PATH
            return command
