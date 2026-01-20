"""UGRO CLI: Main command interface.

Provides command-line interface for GPU cluster orchestration and distributed training.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from .agent import UGROAgent
from .config import load_config

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
):
    """Launch distributed training across cluster."""
    agent: UGROAgent = ctx.obj["agent"]
    
    console.print(f"🚀 Launching Job: {name}", style="bold green")
    
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

if __name__ == "__main__":
    main()
