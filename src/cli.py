#!/usr/bin/env python3
"""
UGRO CLI: Main command interface

Usage:
    ugro health
    ugro launch --name exp1 --epochs 3
    ugro logs exp1
    ugro results exp1
    ugro status
"""

import click
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import UGROAgent
from src.config import load_config

@click.group()
@click.pass_context
def cli(ctx):
    """UGRO: Unified GPU Resource Orchestrator
    
    Personal-scale GPU cluster orchestration tool.
    
    Quick Start:
      ugro health          # Check cluster
      ugro launch          # Start training
      ugro logs <name>     # View logs
      ugro results <name>  # See results
    """
    ctx.ensure_object(dict)
    ctx.obj['agent'] = UGROAgent()
    ctx.obj['config'] = load_config()

@cli.command()
@click.pass_context
def health(ctx):
    """Check cluster health status"""
    agent = ctx.obj['agent']
    
    print("\n🔍 Cluster Health Check")
    print("=" * 60)
    
    health_status = agent.check_cluster_health()
    
    for node_name, status in health_status.items():
        symbol = "✓" if status['healthy'] else "❌"
        print(f"{symbol} {node_name:15} {status['message']}")
    
    print()

@cli.command()
@click.option('--name', required=True, help='Job name')
@click.option('--model', default='unsloth/tinyllama-bnb-4bit', help='Model name')
@click.option('--dataset', default='wikitext', help='Dataset name')
@click.option('--epochs', default=1, type=int, help='Number of epochs')
@click.option('--lr', default=0.0002, type=float, help='Learning rate')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.pass_context
def launch(ctx, name, model, dataset, epochs, lr, verbose):
    """Launch distributed training across cluster"""
    agent = ctx.obj['agent']
    
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
@click.argument('job_name')
@click.option('--rank', default=None, type=int, help='Specific rank')
@click.pass_context
def logs(ctx, job_name, rank):
    """View training logs for a job"""
    agent = ctx.obj['agent']
    agent.display_logs(job_name, rank)

@cli.command()
@click.argument('job_name')
@click.pass_context
def results(ctx, job_name):
    """Show results summary for a job"""
    agent = ctx.obj['agent']
    agent.display_results(job_name)

@cli.command()
@click.pass_context
def status(ctx):
    """Show current cluster status"""
    agent = ctx.obj['agent']
    agent.display_status()

def main():
    """Main entry point"""
    cli(obj={})

if __name__ == '__main__':
    main()