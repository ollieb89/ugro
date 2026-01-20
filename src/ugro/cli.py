"""
UGRO CLI Interface

Command-line interface for UGRO GPU cluster management.
"""

import sys
from pathlib import Path

# Add src to path for imports (needed when called via entry point)
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import cli as src_cli
    cli = src_cli.cli
except ImportError:
    # Fallback if src is not available
    import click
    
    @click.group()
    @click.pass_context
    def cli(ctx):
        """UGRO: Unified GPU Resource Orchestrator"""
        click.echo("UGRO CLI not fully available. Please ensure the package is properly installed.")
        ctx.ensure_object(dict)
    
    # Add basic commands for fallback
    @cli.command()
    def version():
        """Show UGRO version"""
        click.echo("UGRO version 0.1.0")

if __name__ == "__main__":
    cli()
