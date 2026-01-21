# tests/test_cli.py
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_cli_import():
    """Test that CLI can be imported"""
    from ugro.cli import app
    assert app is not None
    assert app.info.name == "UGRO: Unified GPU Resource Orchestrator"

def test_cli_health_command():
    """Test the health command exists"""
    from ugro.cli import app
    commands = [cmd.name for cmd in app.commands.values()]
    assert "health" in commands
    assert "launch" in commands
    assert "status" in commands
