"""SSH utilities for remote execution."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SSHClient:
    """Simple SSH client wrapper"""
    
    def __init__(self, host: str, user: str, port: int = 22):
        """Initialize SSH client
        
        Args:
            host: Remote host address
            user: Username for SSH connection
            port: SSH port (default: 22)
        """
        self.host = host
        self.user = user
        self.port = port
        self.ssh_options = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            '-o', 'ConnectTimeout=10',
            '-o', 'BatchMode=yes'
        ]
    
    def run_command(self, command: str, timeout: int = 30) -> tuple[bool, str, str]:
        """Run command on remote host
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        ssh_cmd = [
            'ssh',
            f'-p{self.port}',
            *self.ssh_options,
            f'{self.user}@{self.host}',
            command
        ]
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"SSH error: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test SSH connection to host
        
        Returns:
            True if connection successful, False otherwise
        """
        success, stdout, stderr = self.run_command('echo "connection_test"', timeout=5)
        return success and "connection_test" in stdout
    
    def copy_file(self, local_path: str, remote_path: str) -> bool:
        """Copy file to remote host
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            
        Returns:
            True if copy successful, False otherwise
        """
        scp_cmd = [
            'scp',
            '-P', str(self.port),
            *self.ssh_options,
            local_path,
            f'{self.user}@{self.host}:{remote_path}'
        ]
        
        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def get_gpu_info(self) -> tuple[bool, dict]:
        """Get GPU information from remote host
        
        Returns:
            Tuple of (success, gpu_info_dict)
        """
        # Try nvidia-smi first
        nvidia_cmd = 'nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'
        success, stdout, stderr = self.run_command(nvidia_cmd, timeout=10)
        
        if success and stdout.strip():
            lines = stdout.strip().split('\n')
            if lines and len(lines[0].split(',')) >= 4:
                name, total_mem, used_mem, util = lines[0].split(',')
                return True, {
                    'name': name.strip(),
                    'memory_total': int(total_mem.strip()),
                    'memory_used': int(used_mem.strip()),
                    'utilization': int(util.strip()),
                    'available': True
                }
        
        # Fallback: check if GPU exists but nvidia-smi failed
        success, _, _ = self.run_command('which nvidia-smi', timeout=5)
        if success:
            return True, {
                'name': 'Unknown GPU',
                'memory_total': 0,
                'memory_used': 0,
                'utilization': 0,
                'available': False
            }
        
        return False, {'available': False}
    
    def check_python_environment(self) -> tuple[bool, dict]:
        """Check Python environment on remote host
        
        Returns:
            Tuple of (success, env_info_dict)
        """
        checks = {}
        
        # Check Python version
        success, stdout, _ = self.run_command('python3 --version', timeout=5)
        checks['python'] = success
        if success:
            checks['python_version'] = stdout.strip()
        
        # Check PyTorch
        success, _, _ = self.run_command('python3 -c "import torch; print(torch.__version__)"', timeout=10)
        checks['pytorch'] = success
        if success:
            checks['pytorch_version'] = stdout.strip()
        
        # Check CUDA availability
        success, stdout, _ = self.run_command('python3 -c "import torch; print(torch.cuda.is_available())"', timeout=10)
        checks['cuda'] = success and 'True' in stdout.strip()
        
        return all([checks['python'], checks['pytorch']]), checks
