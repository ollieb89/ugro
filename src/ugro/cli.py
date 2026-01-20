"""UGRO CLI: Main command interface.

Provides command-line interface for GPU cluster orchestration and distributed training.

Usage:
    ugro health                          # Check cluster health
    ugro launch --name exp1 --epochs 3   # Launch training job
    ugro logs exp1                       # View job logs
    ugro results exp1                    # Show job results
    ugro status                          # Display cluster status
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

# Import from same package using relative imports
from .agent import UGROAgent
from .config import load_config


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """UGRO: Unified GPU Resource Orchestrator.
    
    Personal-scale GPU cluster orchestration tool for distributed training.
    
    Quick Start:
        ugro health          # Check cluster health
        ugro test-setup      # Verify setup logic (w1, w2, etc.)
        ugro launch          # Start training job
        ugro logs <name>     # View training logs
        ugro results <name>  # Display results
    """
    ctx.ensure_object(dict)
    config = load_config()
    ctx.obj["config"] = config
    ctx.obj["agent"] = UGROAgent(config=config)


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check cluster health status."""
    agent: UGROAgent = ctx.obj["agent"]
    
    click.echo("\n🔍 Cluster Health Check")
    click.echo("=" * 60)
    
    health_status: Mapping[str, Any] = agent.check_cluster_health()
    
    for name, status in health_status.items():
        symbol = "✓" if status["healthy"] else "❌"
        click.echo(f"{symbol} {name}: {status['message']}")
    
    # Test worker operations as shown in documentation
    workers = agent.cluster.get_all_workers()
    if workers:
        click.echo("\n🔧 Worker Operations Test:")
        worker = workers[0]['name']
        success, stdout, stderr = agent.cluster.execute_on_worker(worker, 'echo "Hello from worker"')
        if success:
            click.echo(f'Worker command: {success} - {stdout.strip()}')
        else:
            click.echo(f'Worker command failed: {stderr.strip()}')
    
    click.echo()


@cli.command()
@click.option("--name", required=True, help="Job name")
@click.option("--model", default="unsloth/tinyllama-bnb-4bit", help="Model name")
@click.option("--dataset", default="wikitext", help="Dataset name")
@click.option("--epochs", default=1, type=int, help="Number of epochs")
@click.option("--lr", default=0.0002, type=float, help="Learning rate")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def launch(
    ctx: click.Context,
    name: str,
    model: str,
    dataset: str,
    epochs: int,
    lr: float,
    verbose: bool,
) -> None:
    """Launch distributed training across cluster."""
    agent: UGROAgent = ctx.obj["agent"]
    
    success = agent.launch_training(
        job_name=name,
        model=model,
        dataset=dataset,
        epochs=epochs,
        learning_rate=lr,
        verbose=verbose,
    )
    
    sys.exit(0 if success else 1)


@cli.command()
@click.argument("job_name")
@click.option("--rank", default=None, type=int, help="Specific rank to view")
@click.pass_context
def logs(ctx: click.Context, job_name: str, rank: int | None) -> None:
    """View training logs for a job."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_logs(job_name, rank)


@cli.command()
@click.argument("job_name")
@click.pass_context
def results(ctx: click.Context, job_name: str) -> None:
    """Show results summary for a job."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_results(job_name)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current cluster status."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_status()


@cli.command()
@click.argument("workers", nargs=-1)
@click.pass_context
def test_setup(ctx: click.Context, workers: tuple[str, ...]) -> None:
    """Test setup and connectivity for specific workers (e.g., w1, w2, gpu1)."""
    agent: UGROAgent = ctx.obj["agent"]
    
    click.echo("\n🧪 UGRO Setup Verification")
    click.echo("=" * 60)
    
    # 0. Directory Structure Check
    click.echo("\n📁 Directory Structure Check")
    click.echo("-" * 60)
    required_dirs = ["config", "src", "scripts", "data", "logs"]
    project_root = Path.cwd()
    for d in required_dirs:
        path = project_root / d
        status = "✅" if path.exists() else "❌"
        click.echo(f"{status} {d:10} ({path})")
    
    # Resolve worker shorthands
    target_workers: list[str] = []
    all_workers = agent.cluster.get_all_workers()
    
    if not workers:
        target_workers = [w["name"] for w in all_workers]
    else:
        for w_id in workers:
            w_id = w_id.lower()
            if w_id.startswith("w") and w_id[1:].isdigit():
                # Map w1 -> rank 1, w2 -> rank 2
                rank = int(w_id[1:])
                worker = agent.cluster.get_worker_by_rank(rank)
                if worker:
                    target_workers.append(worker["name"])
                else:
                    click.echo(f"⚠️  Worker with rank {rank} (from {w_id}) not found.")
            else:
                # Direct name match
                worker = agent.cluster.get_worker_by_name(w_id)
                if worker:
                    target_workers.append(worker["name"])
                else:
                    click.echo(f"⚠️  Worker '{w_id}' not found.")

    if not target_workers:
        click.echo("❌ No valid workers found to test.")
        sys.exit(1)

    # Perform checks
    click.echo(f"Testing connectivity for: {', '.join(target_workers)}")
    click.echo("-" * 60)

    for name in target_workers:
        worker = agent.cluster.get_worker_by_name(name)
        if not worker:
            continue
            
        click.echo(f"\n📡 Node: {name} ({worker['ip']})")
        
        # 1. SSH Connectivity
        ssh_client = agent.cluster.ssh_clients.get(name)
        if not ssh_client:
            click.echo("  ❌ SSH Client not initialized")
            continue
            
        success, stdout, stderr = ssh_client.run_command('echo "connection_test"', timeout=5)
        if success and "connection_test" in stdout:
            click.echo("  ✅ SSH Connectivity: OK")
        else:
            click.echo(f"  ❌ SSH Connectivity: FAILED")
            if stderr:
                click.echo(f"     Error: {stderr.strip()}")
            continue

        # 2. GPU Availability
        success, gpu_info = ssh_client.get_gpu_info()
        if success:
            click.echo(f"  ✅ GPU Found: {gpu_info['name']} ({gpu_info['memory_total']}MB)")
        else:
            click.echo("  ❌ GPU Check: FAILED (nvidia-smi not working)")

        # 3. Environment Check
        success, env_info = ssh_client.check_python_environment()
        if success:
            click.echo(f"  ✅ Python: {env_info.get('python_version', 'Unknown')}")
            click.echo(f"  ✅ PyTorch: {env_info.get('pytorch_version', 'Unknown')} (CUDA: {env_info.get('cuda', False)})")
        else:
            click.echo("  ❌ Environment: FAILED")
            if 'error' in env_info:
                click.echo(f"     Error: {env_info['error']}")
            elif 'pytorch_error' in env_info:
                click.echo(f"     PyTorch Error: {env_info['pytorch_error']}")

    click.echo("\n" + "=" * 60)
    click.echo("✨ Setup check complete.")


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
