# tests/test_cli_wandb_security.py
import pytest
from click.testing import CliRunner
from ugro.cli import hpo

def test_wandb_project_validation():
    runner = CliRunner()
    
    # Test invalid project name
    result = runner.invoke(hpo, [
        'sweep',
        '--wandb-project', 'invalid project!',
        '--dry-run'
    ])
    assert result.exit_code != 0
    assert 'Invalid project name' in result.output
