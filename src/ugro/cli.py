"""UGRO CLI: Main command interface.

Provides command-line interface for GPU cluster orchestration and distributed training.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Configure logging to suppress noisy external libraries
import logging
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

from .agent import UGROAgent
from .config import load_config, QueueType
from .database import Database
# from .queue import JobQueue # Legacy
from .queues import SQLiteJobQueue, RedisJobQueue, Job, JobStatus, JobPriority, JobResources
from .scheduler import Scheduler, ResourceTracker

# Initialize Typer app and Rich console
app = typer.Typer(help="UGRO: Unified GPU Resource Orchestrator")
console = Console()
error_console = Console(stderr=True, style="bold red")

State = dict[str, Any]

@app.callback()
def context_callback(ctx: typer.Context):
    """Initialize global context."""
    try:
        config = load_config()
        ctx.ensure_object(dict)
        if "database" not in ctx.obj:
             ctx.obj["database"] = Database()
        if "database" not in ctx.obj:
             ctx.obj["database"] = Database()
        if "queue" not in ctx.obj:
             cluster_conf = config.get("cluster", {})
             # Handle both dict access (legacy load_config return) and object access if we improved it
             # Current load_config returns dict. config.py models are used inside UGROAgent, 
             # but here load_config returns data.
             # We should probably parse it into AppConfig to be safe, but legacy code uses dict.
             # Let's check keys manually or use the model defaults if missing.
             
             q_conf = cluster_conf.get("queue", {})
             q_type = q_conf.get("type", "sqlite")
             
             if q_type == "redis":
                 host = q_conf.get("redis_host", "localhost")
                 port = q_conf.get("redis_port", 6379)
                 db = q_conf.get("redis_db", 0)
                 password = q_conf.get("redis_password", None)
                 
                 try:
                     ctx.obj["queue"] = RedisJobQueue(host=host, port=port, db=db, password=password)
                 except ImportError:
                     error_console.print("❌ Redis support requires redis-py. Install with `pip install redis` or switch to sqlite.")
                     raise typer.Exit(code=1)
                 except Exception as e:
                     error_console.print(f"❌ Failed to connect to Redis: {e}")
                     raise typer.Exit(code=1)
             else:
                 # Default to sqlite
                 ctx.obj["queue"] = SQLiteJobQueue(db_path=str(ctx.obj["database"].db_path))

        ctx.obj["config"] = config
        ctx.obj["agent"] = UGROAgent(config=config)
    except Exception as e:
        error_console.print(f"Failed to initialize UGRO: {e}")
        # typer doesn't have a clean way to exit from callback without traceback in some versions,
        # but asking for help shows we failed.
        # raising Exit is best.
        raise typer.Exit(code=1)


@app.command()
def health(ctx: typer.Context):
    """Check cluster health status."""
    agent: UGROAgent = ctx.obj["agent"]
    
    console.print("\n🔍 Cluster Health Check", style="bold blue")
    console.print("=" * 60)
    
    # Prefer cached state from daemon if it looks recent
    cached_state = agent.cluster_state_manager.get_state()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Node")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Last Check")
    
    for name, node in cached_state.nodes.items():
        # Map status string to emoji/style
        status_map = {
            "healthy": ("✓", "green"),
            "degraded": ("!", "yellow"),
            "unhealthy": ("❌", "red"),
            "busy": ("⏳", "blue"),
            "available": ("✓", "green")
        }
        symbol, style = status_map.get(node.status, ("?", "white"))
        
        # Get score if available in node metadata (HealthMonitor updates it)
        # Note: NodeState might need health_score field to be truly integrated
        score = getattr(node, "health_score", 100.0) 
        last_check = getattr(node, "last_check", "N/A")
        
        table.add_row(
            name, 
            Text(f"{symbol} {node.status}", style=style),
            f"{score:.1f}",
            str(last_check)
        )
        
    console.print(table)
    
    # Simple direct connectivity check for UX
    console.print("\nRunning quick connectivity test...", style="italic dim")

    # Access cluster directly for a quick ping test
    # Note: For strict async CLI, we would use an async command wrapper.
    # UGROAgent methods are synchronous wrappers over async logic (via asyncio.run),
    # so we can call them directly in sync Typer commands for now.
    
    console.print()


@app.command()
def launch(
    ctx: typer.Context,
    name: Annotated[str, typer.Option("--name", "-n", help="Job name")] = "job_1",
    model: Annotated[str, typer.Option("--model", "-m", help="Model name")] = "unsloth/tinyllama-bnb-4bit",
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset name")] = "wikitext",
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of epochs")] = 1,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 0.0002,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
    now: Annotated[bool, typer.Option("--now", help="Launch immediately, bypassing queue")] = False,
):
    """Launch distributed training across cluster."""
    agent: UGROAgent = ctx.obj["agent"]
    queue: JobQueue = ctx.obj["queue"]
    
    # Configuration dictionary for the job
    job_config = {
        "model": model,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": lr,
        "verbose": verbose
    }

    if not now:
        # Enqueue the job
        job = Job(
            name=name,
            command=f"ugro launch --now --name {name} --model {model} --dataset {dataset} --epochs {epochs} --lr {lr} {'--verbose' if verbose else ''}",
            priority=JobPriority.NORMAL,
            resources=JobResources(gpu_count=1), # Default 1 GPU
            metadata=job_config
        )
        job_id = queue.submit(job)
        console.print(f"✅ Job enqueued successfully!", style="bold green")
        console.print(f"   Job ID: {job_id}")
        console.print(f"   Model: {model}")
        console.print(f"   Dataset: {dataset}")
        console.print(f"\nRun [bold]ugro queue list[/bold] to see pending jobs.")
        console.print(f"Run [bold]ugro run-worker[/bold] to process the queue.")
        return

    # Immediate launch logic
    console.print(f"🚀 Launching Job: {name} (Immediate Mode)", style="bold green")
    
    success = agent.launch_training(
        job_name=name,
        model=model,
        dataset=dataset,
        epochs=epochs,
        learning_rate=lr,
        verbose=verbose,
    )
    
    if not success:
        error_console.print("❌ Launch failed")
        raise typer.Exit(code=1)


@app.command()
def logs(
    ctx: typer.Context,
    job_name: Annotated[str, typer.Argument(help="Job name")],
    rank: Annotated[Optional[int], typer.Option(help="Specific rank to view")] = None
):
    """View training logs for a job."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_logs(job_name, rank)


# Queue Management Group
queue_app = typer.Typer(help="Manage job queue")
app.add_typer(queue_app, name="queue")

@queue_app.command("list")
def queue_list(ctx: typer.Context, limit: int = 10):
    """List recent jobs in the queue."""
    queue = ctx.obj["queue"] # Use dynamic type
    jobs = queue.list_jobs(limit=limit)
    
    if not jobs:
        console.print("No jobs found.", style="yellow")
        return
        
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Created At")
    
    for job in jobs:
        status_style = {
            JobStatus.PENDING: "yellow",
            JobStatus.RUNNING: "blue",
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red",
            JobStatus.CANCELLED: "dim white"
        }.get(job.status, "white")
        
        table.add_row(
            job.id[:8], # Short ID
            f"[{status_style}]{job.status}[/{status_style}]",
            job.metadata.get("model", "N/A"),
            job.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    console.print(table)

@queue_app.command("inspect")
def queue_inspect(ctx: typer.Context, job_id: str):
    """Inspect a specific job."""
    queue = ctx.obj["queue"]
    # Provide full ID or try to find by short ID? 
    # For now assume full ID or we Implement prefix search in queue
    job = queue.get_job(job_id)
    
    if not job:
        # Fallback: list all and check prefixes (inefficient but helpful for CLI)
        all_jobs = queue.list_jobs(limit=100)
        for j in all_jobs:
            if j.id.startswith(job_id):
                job = j
                break
    
    if not job:
        error_console.print(f"Job {job_id} not found.")
        raise typer.Exit(code=1)
        
    console.print(f"\n🔍 Job Details: {job.id}", style="bold blue")
    console.print(f"Status: {job.status}")
    console.print(f"Priority: {job.priority}")
    console.print(f"Command: {job.command}")
    console.print(f"Created: {job.created_at}")
    if job.started_at:
        console.print(f"Started: {job.started_at}")
    if job.completed_at:
        console.print(f"Completed: {job.completed_at}")
    
    console.print("\nMetadata:", style="bold")
    console.print(job.metadata)

@queue_app.command("cancel")
def queue_cancel(ctx: typer.Context, job_id: str):
    """Cancel a pending job."""
    queue = ctx.obj["queue"]
    
    # Try find by prefix
    target_id = job_id
    if len(job_id) < 36:
         all_jobs = queue.list_jobs(limit=100)
         for j in all_jobs:
            if j.id.startswith(job_id):
                target_id = j.id
                break

    if queue.cancel(target_id):
        console.print(f"✅ Job {target_id} cancelled.", style="green")
    else:
        error_console.print(f"❌ Failed to cancel {job_id}. Job may be running, completed or not found.")

@app.command("run-worker")
def run_worker(
    ctx: typer.Context, 
    loop_interval: Annotated[float, typer.Option("--interval", "-i", help="Polling interval")] = 5.0
):
    """Start a worker process to consume jobs from the queue."""
    # New Implementation using Scheduler
    queue = ctx.obj["queue"]
    config = ctx.obj["config"]
    
    console.print("👷 Starting Worker...", style="bold green")
    
    tracker = ResourceTracker(cluster_config=config.get('cluster', {}))
    scheduler = Scheduler(queue=queue, resource_tracker=tracker, poll_interval=loop_interval)
    
    scheduler.loop()


# Daemon Management Group
daemon_app = typer.Typer(help="Manage UGRO background daemons")
app.add_typer(daemon_app, name="daemon")

@daemon_app.command("start")
def daemon_start():
    """Start the health monitor daemon."""
    import subprocess
    console.print("🚀 Starting UGRO Monitor Daemon...", style="bold green")
    try:
        # Check if systemd service exists (check user session)
        res = subprocess.run(["systemctl", "--user", "list-unit-files", "ugro-monitor.service"], capture_output=True, text=True)
        if "ugro-monitor.service" in res.stdout:
            subprocess.run(["systemctl", "--user", "start", "ugro-monitor.service"], check=False)
            console.print("✓ Attempted start via systemd (user session)")
        else:
            # Fallback to direct process starting in background
            log_file = Path("logs/monitor_daemon.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            script_path = Path(__file__).parent.parent.parent / "scripts" / "monitor_daemon.py"
            with open(log_file, "a") as f:
                subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=f, stderr=f,
                    cwd=Path.cwd(),
                    start_new_session=True
                )
            console.print(f"✓ Started background process (logs: {log_file})")
    except Exception as e:
        error_console.print(f"❌ Failed to start daemon: {e}")

@daemon_app.command("stop")
def daemon_stop():
    """Stop the health monitor daemon."""
    import subprocess
    console.print("🛑 Stopping UGRO Monitor Daemon...", style="bold red")
    try:
        subprocess.run(["systemctl", "--user", "stop", "ugro-monitor.service"], check=False)
        
        # Kill any manual processes using psutil
        try:
            import psutil
            killed = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('monitor_daemon.py' in arg for arg in cmdline):
                        proc.terminate()
                        proc.wait(timeout=5)
                        killed = True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
            
            if killed:
                console.print("✓ Stopped running daemon process")
        except ImportError:
            console.print(" ⚠️ psutil not available, systemd stop attempted only")
            
        console.print("✓ Stopped")
    except Exception as e:
        error_console.print(f"❌ Error stopping daemon: {e}")

@daemon_app.command("status")
def daemon_status():
    """Show daemon status."""
    try:
        import psutil
        
        # Look for monitor_daemon.py process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('monitor_daemon.py' in arg for arg in cmdline):
                    console.print(f"🟢 UGRO Monitor Daemon is [bold green]RUNNING[/] (PID: {proc.info['pid']})")
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        console.print("🔴 UGRO Monitor Daemon is [bold red]STOPPED[/]")
    except ImportError:
        console.print("⚠️ psutil not available, cannot check daemon status")
        console.print("💡 Install with: pixi add psutil")

@app.command()
def monitor(
    ctx: typer.Context,
    job_name: Annotated[str, typer.Argument(help="Job name")],
    refresh: Annotated[float, typer.Option("--refresh", "-r", help="Refresh interval in seconds")] = 2.0
):
    """Show live training dashboard."""
    agent: UGROAgent = ctx.obj["agent"]
    from .dashboard import LiveDashboard
    
    dashboard = LiveDashboard(agent, job_id=job_name, refresh_interval=refresh)
    try:
        asyncio.run(dashboard.run())
    except Exception as e:
        error_console.print(f"Error in dashboard: {e}")


@app.command()
def results(
    ctx: typer.Context,
    job_name: Annotated[str, typer.Argument(help="Job name")]
):
    """Show results summary for a job."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_results(job_name)


@app.command()
def status(ctx: typer.Context):
    """Show current cluster status."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_status()


@app.command()
def test_setup(
    ctx: typer.Context,
    workers: Annotated[Optional[list[str]], typer.Argument(help="Specific workers to test")] = None
):
    """Verify cluster setup and connectivity."""
    agent: UGROAgent = ctx.obj["agent"]
    
    console.print("\n🧪 UGRO Setup Verification", style="bold cyan")
    console.print("=" * 60)
    
    # Check Directories via pathlib for quick feedback
    console.print("\n📁 Directory Checks:", style="bold")
    paths = ["config", "src", "scripts", "data", "logs"]
    cwd = Path.cwd()
    for p in paths:
        path = cwd / p
        if path.exists():
            console.print(f"  ✓ {p}/", style="green")
        else:
            console.print(f"  ❌ {p}/ (Missing)", style="red")
    
    # Resolve workers to test
    if not workers:
        target_workers = agent.cluster.get_all_workers()
    else:
        # Simple filter logic
        all_workers = agent.cluster.get_all_workers()
        # Filter where name is in workers OR 'w{rank}' pattern
        # The argument 'workers' is a list of strings
        target_workers = []
        for w_arg in workers:
            found = False
            for w_conf in all_workers:
                if w_conf['name'] == w_arg:
                    target_workers.append(w_conf)
                    found = True
                    break
                # Try rank map if arg looks like w1
                if w_arg.lower().startswith('w') and w_arg[1:].isdigit():
                    rank = int(w_arg[1:])
                    if w_conf.get('rank') == rank:
                        target_workers.append(w_conf)
                        found = True
                        break
            if not found:
                console.print(f"Warning: Worker arg '{w_arg}' moved to unknown.", style="yellow")

    if not target_workers:
        console.print("No workers found to test.", style="yellow")
        return

    console.print(f"\n📡 Connectivity Check ({len(target_workers)} nodes):", style="bold")
    
    # We will run this synchronously for now to keep the CLI simple,
    # as existing agent logic is wrapped.
    for worker in target_workers:
        name = worker['name']
        ip = worker['ip']
        console.print(f"\nNode: {name} ({ip})", style="bold underline")
        
        ssh_client = agent.cluster.ssh_clients.get(name)
        if not ssh_client:
             console.print("  ❌ Client not initialized", style="red")
             continue
             
        # 1. SSH
        # ssh_client.run_command is synchronous (and potentially blocking)
        # but for test-setup it's acceptable.
        success, out, err = ssh_client.run_command('echo "OK"', timeout=5)
        if success:
             console.print("  ✓ SSH: OK", style="green")
        else:
             console.print(f"  ❌ SSH: Fail ({err.strip()})", style="red")
             continue
             
        # 2. GPU
        success, gpu = ssh_client.get_gpu_info()
        if success:
            console.print(f"  ✓ GPU: {gpu.get('name')} ({gpu.get('memory_total')}MB)", style="green")
        else:
            console.print("  ❌ GPU: detection failed", style="red")



def main():
    app()


# =============================================================================
# HPO (Hyperparameter Optimization) Command Group
# =============================================================================

hpo_app = typer.Typer(help="Hyperparameter optimization (HPO) commands")
app.add_typer(hpo_app, name="hpo")


@hpo_app.command("sweep")
def hpo_sweep(
    ctx: typer.Context,
    study_name: Annotated[str, typer.Option("--study-name", "-s", help="Unique study name")] = "ugro-sweep",
    search_space: Annotated[str, typer.Option("--search-space", "-c", help="Path to search space YAML")] = "config/llama_lora_hpo.yaml",
    n_trials: Annotated[int, typer.Option("--n-trials", "-n", help="Number of trials")] = 50,
    parallel_jobs: Annotated[int, typer.Option("--parallel-jobs", "-j", help="Concurrent trials")] = 4,
    algorithm: Annotated[str, typer.Option("--algorithm", "-a", help="Algorithm: tpe, asha, hyperband, pbt")] = "tpe",
    scheduler: Annotated[Optional[str], typer.Option("--scheduler", help="Scheduler: asha, hyperband, pbt")] = None,
    ray_address: Annotated[Optional[str], typer.Option("--ray-address", help="Ray cluster address")] = None,
    ray_gpu_per_trial: Annotated[float, typer.Option("--ray-gpu", help="GPU fraction per trial")] = 1.0,
    storage_backend: Annotated[str, typer.Option("--storage", help="Optuna storage URI")] = "sqlite:///ugro_hpo.db",
    tracking_uri: Annotated[Optional[str], typer.Option("--tracking-uri", help="MLflow tracking URI")] = None,
    wandb_project: Annotated[Optional[str], typer.Option("--wandb-project", help="W&B project name")] = None,
    export_best: Annotated[Optional[str], typer.Option("--export-best", help="Export best config to YAML")] = None,
    save_trials_csv: Annotated[Optional[str], typer.Option("--save-trials", help="Save all trials to CSV")] = None,
    model_id: Annotated[str, typer.Option("--model", "-m", help="Model ID for training")] = "unsloth/tinyllama-bnb-4bit",
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset name")] = "wikitext",
    max_steps: Annotated[int, typer.Option("--max-steps", help="Max training steps per trial")] = 500,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate config without running")] = False,
):
    """Execute distributed hyperparameter optimization sweep.
    
    Uses Ray Tune with Optuna's multivariate TPE sampler for efficient
    hyperparameter search with optional early stopping via ASHA scheduler.
    
    Example:
        ugro hpo sweep --study-name llama-lora --n-trials 100 --parallel-jobs 8
    """
    from pathlib import Path
    
    # Validate W&B project name if provided
    if wandb_project is not None:
        try:
            from .hpo.security import validate_project_name, mask_api_key
            if not validate_project_name(wandb_project):
                error_console.print(f"❌ Invalid W&B project name '{wandb_project}'")
                error_console.print("   Project names must be alphanumeric with hyphens/underscores")
                error_console.print("   Length: 1-128 characters")
                raise typer.Exit(code=1)
            
            # Mask API key in logs if present
            api_key = os.environ.get('WANDB_API_KEY')
            if api_key:
                console.print(f"   W&B Project: {wandb_project}", style="blue")
                console.print(f"   API Key: {mask_api_key(api_key)}", style="dim blue")
        except ImportError:
            # Security module not available, skip validation
            console.print("   ⚠️  W&B security validation not available", style="yellow")
    
    try:
        from ugro.hpo.config import HPOConfig, OptimizerAlgorithm
        from ugro.hpo.search_space import (
            load_search_space_yaml,
            parse_parameter_bounds,
            parse_objectives,
            parse_constraints,
        )
        from ugro.hpo.objective import LoRAFinetuningObjective
        from ugro.hpo.optimizer import UGROOptimizer
    except ImportError as e:
        error_console.print(f"❌ HPO dependencies not installed: {e}")
        error_console.print("💡 Install with: pixi add -e hpo 'ray[tune]' optuna mlflow")
        raise typer.Exit(code=1)
    
    # Load and validate search space
    search_space_path = Path(search_space)
    if not search_space_path.exists():
        error_console.print(f"❌ Search space not found: {search_space}")
        raise typer.Exit(code=1)
    
    console.print(f"📊 Loading search space: {search_space}", style="bold blue")
    config_dict = load_search_space_yaml(search_space)
    bounds = parse_parameter_bounds(config_dict)
    objectives = parse_objectives(config_dict)
    constraints = parse_constraints(config_dict)
    
    console.print(f"   Parameters: {len(bounds)}")
    for b in bounds:
        console.print(f"      • {b.name}: {b.type} [{b.min} - {b.max}]" if b.type != "categorical" else f"      • {b.name}: {b.choices}")
    
    if constraints:
        console.print(f"   Constraints: {len(constraints)}")
        for c in constraints:
            console.print(f"      • {c}")
    
    if dry_run:
        console.print("\n✅ Dry run complete - config is valid", style="green")
        return
    
    # Create HPO config
    hpo_config = HPOConfig(
        study_name=study_name,
        search_space=bounds,
        objectives=objectives,
        algorithm=OptimizerAlgorithm(algorithm),
        scheduler_type=scheduler,
        n_trials=n_trials,
        parallel_jobs=parallel_jobs,
        ray_address=ray_address,
        ray_gpu_per_trial=ray_gpu_per_trial,
        storage_backend=storage_backend,
        tracking_uri=tracking_uri,
        wandb_project=wandb_project,
        export_best=export_best,
        save_trials_csv=save_trials_csv,
    )
    
    # Create objective function
    console.print(f"\n🎯 Creating objective: {model_id} on {dataset}", style="bold blue")
    objective = LoRAFinetuningObjective(
        model_id=model_id,
        dataset_name=dataset,
        max_steps=max_steps,
        use_mlflow=tracking_uri is not None,
        use_wandb=wandb_project is not None,
        constraints=constraints,
        objectives=[{"name": obj.name, "direction": obj.direction} for obj in objectives],
        parameter_bounds=bounds,
    )
    
    # Run optimization
    console.print(f"\n🚀 Starting HPO: {n_trials} trials, {parallel_jobs} parallel", style="bold green")
    console.print(f"   Storage: {storage_backend}")
    if tracking_uri:
        console.print(f"   Tracking: {tracking_uri}")
    if wandb_project:
        console.print(f"   W&B Project: {wandb_project}")
    
    optimizer = UGROOptimizer(hpo_config, objective)
    
    try:
        results = optimizer.optimize()
        
        console.print("\n✅ HPO Complete!", style="bold green")
        console.print(f"\n📈 Best Results:")
        console.print(f"   Trials: {results.get('n_trials', 'N/A')}")
        
        best_config = results.get("best_config", {})
        console.print("\n   Best Parameters:")
        for k, v in best_config.items():
            console.print(f"      {k}: {v}")
        
        best_metrics = results.get("best_metrics", {})
        console.print("\n   Best Metrics:")
        for k, v in best_metrics.items():
            if isinstance(v, float):
                console.print(f"      {k}: {v:.6f}")
            else:
                console.print(f"      {k}: {v}")
        
        if export_best:
            console.print(f"\n   Config exported to: {export_best}")
        
    except Exception as e:
        error_console.print(f"❌ HPO failed: {e}")
        raise typer.Exit(code=1)


@hpo_app.command("analyze")
def hpo_analyze(
    storage_backend: Annotated[str, typer.Option("--storage", "-s", help="Optuna storage URI")] = "sqlite:///ugro_hpo.db",
    study_name: Annotated[str, typer.Option("--study-name", "-n", help="Study name to analyze")] = "ugro-sweep",
    output_dir: Annotated[Optional[str], typer.Option("--output", "-o", help="Output directory for charts")] = None,
):
    """Analyze HPO study results with visualizations.
    
    Generates parameter importance analysis, trial progression plots,
    and optimization history visualization.
    """
    try:
        from ugro.hpo.analysis import analyze_hpo_results
    except ImportError as e:
        error_console.print(f"❌ Analysis dependencies not installed: {e}")
        error_console.print("💡 Install with: pixi add optuna matplotlib pandas")
        raise typer.Exit(code=1)
    
    console.print(f"📊 Analyzing study: {study_name}", style="bold blue")
    
    try:
        results = analyze_hpo_results(
            storage_backend=storage_backend,
            study_name=study_name,
            output_dir=output_dir,
        )
        
        console.print("\n✅ Analysis complete!", style="green")
        if output_dir:
            console.print(f"   Charts saved to: {output_dir}")
        
    except Exception as e:
        error_console.print(f"❌ Analysis failed: {e}")
        raise typer.Exit(code=1)


@hpo_app.command("export-best")
def hpo_export_best(
    storage_backend: Annotated[str, typer.Option("--storage", "-s", help="Optuna storage URI")] = "sqlite:///ugro_hpo.db",
    study_name: Annotated[str, typer.Option("--study-name", "-n", help="Study name")] = "ugro-sweep",
    output: Annotated[str, typer.Option("--output", "-o", help="Output YAML path")] = "config/best_params.yaml",
):
    """Export best hyperparameters from study to YAML.
    
    Useful for extracting optimal configuration after HPO completes.
    """
    try:
        from ugro.hpo.analysis import export_best_config
    except ImportError as e:
        error_console.print(f"❌ Export dependencies not installed: {e}")
        raise typer.Exit(code=1)
    
    console.print(f"📤 Exporting best config from: {study_name}", style="bold blue")
    
    try:
        best_params = export_best_config(
            storage_backend=storage_backend,
            study_name=study_name,
            output_path=output,
        )
        
        console.print(f"\n✅ Exported to: {output}", style="green")
        console.print("\nBest Parameters:")
        for k, v in best_params.items():
            console.print(f"   {k}: {v}")
        
    except Exception as e:
        error_console.print(f"❌ Export failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main()
