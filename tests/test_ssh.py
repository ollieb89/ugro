# tests/test_ssh.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_ssh_client_creation():
    """Test SSH client can be created"""
    from ugro.ssh_utils import SSHClient
    
    client = SSHClient(
        host="192.168.1.101",
        user="testuser",
        port=22
    )
    
    assert client.host == "192.168.1.101"
    assert client.user == "testuser"
    assert client.port == 22
    assert client.ssh_executable is not None

def test_ssh_command_building():
    """Test SSH command building"""
    from ugro.ssh_utils import SSHClient
    
    client = SSHClient(
        host="192.168.1.101",
        user="testuser",
        env_command="pixi run"
    )
    
    # Test that SSH options are configured
    assert 'StrictHostKeyChecking=no' in client.ssh_options
    assert 'ConnectTimeout=10' in client.ssh_options
    assert client.env_command == "pixi run"
