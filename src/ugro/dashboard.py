"""Live CLI dashboard for UGRO training monitoring."""

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if TYPE_CHECKING:
    from .agent import UGROAgent

class LiveDashboard:
    """Rich-based live dashboard for training monitoring."""
    
    def __init__(self, agent: "UGROAgent", job_id: str, refresh_interval: float = 2.0):
        self.agent = agent
        self.job_id = job_id
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.start_time = time.time()

    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10),
        )
        layout["main"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="info", ratio=1),
        )
        return layout

    def get_header(self) -> Panel:
        """Create header panel."""
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        
        title = Text.from_markup(f"[bold blue]UGRO Training Monitor[/] - [cyan]{self.job_id}[/]")
        status = Text("Status: RUNNING", style="bold green")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        grid.add_row(title, status, now)
        return Panel(grid, style="white on blue")

    def get_metrics_table(self, summary: dict[str, Any]) -> Panel:
        """Create metrics table panel."""
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Rank", style="cyan")
        table.add_column("Step", justify="right")
        table.add_column("Loss", justify="right", style="magenta")
        table.add_column("GPU Util", justify="right", style="green")
        table.add_column("GPU Mem", justify="right", style="blue")
        
        per_rank = summary.get("per_rank_stats", {})
        if not per_rank:
            table.add_row("No metrics available yet...", "", "", "", "")
        else:
            # Sort by rank
            for rank_id in sorted(per_rank.keys()):
                stats = per_rank[rank_id]
                table.add_row(
                    rank_id,
                    str(stats.get("step", "-")),
                    f"{stats.get('loss', 0.0):.4f}" if stats.get('loss') is not None else "-",
                    f"{stats.get('gpu_util', 0.0):.1f}%" if stats.get('gpu_util') is not None else "-",
                    f"{stats.get('gpu_mem_used_gb', 0.0):.1f} GB" if stats.get('gpu_mem_used_gb') is not None else "-"
                )
        
        return Panel(table, title="[bold]Per-Rank Metrics[/]", border_style="cyan")

    def get_info_panel(self, summary: dict[str, Any]) -> Panel:
        """Create job information panel."""
        text = Text()
        text.append(f"Job ID: {summary['job_id']}\n", style="bold")
        text.append(f"Steps: {summary['total_steps']}\n")
        text.append(f"Final Loss: {summary['final_loss']:.4f}\n" if summary.get('final_loss') else "Final Loss: -\n")
        text.append(f"Throughput: {summary['avg_throughput']:.1f} tok/s\n", style="yellow")
        
        duration = summary.get("duration", 0.0)
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        text.append(f"Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n")
        
        return Panel(text, title="[bold]Job Info[/]", border_style="green")

    def get_logs_panel(self, job_id: str) -> Panel:
        """Create logs panel with last few lines of training log."""
        try:
            # Assumes logs are in master's job_dir/logs/rank_0.log or similar
            # For simplicity, we'll try to find a combined log or rank 0 log
            log_dir = self.agent._result_aggregator.jobs_dir / job_id / "logs"
            log_files = list(log_dir.glob("*.log"))
            
            content = ""
            if log_files:
                # Read last 10 lines of the first log file found
                with open(log_files[0], "r", encoding="utf-8") as f:
                    lines = f.readlines()[-8:]
                    content = "".join(lines)
            else:
                content = "Wait for logs..."
        except Exception:
            content = "Could not load logs."
            
        return Panel(Text(content, style="dim"), title="[bold]Latest Logs[/]", border_style="white")

    async def run(self):
        """Run the live dashboard."""
        layout = self.create_layout()
        
        with Live(layout, refresh_per_second=1/self.refresh_interval, screen=True) as live:
            try:
                while True:
                    # Run IO-bound summary fetch in thread if not async, 
                    # but UGRO metadata is local files, so it's fine.
                    summary = self.agent._result_aggregator.get_job_summary(self.job_id)
                    
                    layout["header"].update(self.get_header())
                    layout["metrics"].update(self.get_metrics_table(summary))
                    layout["info"].update(self.get_info_panel(summary))
                    layout["footer"].update(self.get_logs_panel(self.job_id))
                    
                    await asyncio.sleep(self.refresh_interval)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
