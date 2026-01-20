"""SSH utilities for remote execution."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SSHClient:
    """Simple SSH client wrapper"""
    
    def __init__(self, host: str, user: str, port: int = 22, env_command: str | None = None, home_dir: str | None = None, project_dir: str | None = None):
        """Initialize SSH client
        
        Args:
            host: Remote host address
            user: Username for SSH connection
            port: SSH port (default: 22)
            env_command: Command to prefix for environment activation (e.g., 'pixi run')
            home_dir: Remote home directory
            project_dir: Remote project directory where pixi.toml is located
        """
        self.host = host
        self.user = user
        self.port = port
        self.env_command = env_command
        self.home_dir = home_dir
        self.project_dir = project_dir
        self.ssh_executable = self._find_ssh_executable()
        self.ssh_options = [
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            '-o', 'ConnectTimeout=10',
            '-o', 'BatchMode=yes'
        ]
    
    async def run_command_async(self, command: str, timeout: int = 30, use_env: bool = True) -> tuple[bool, str, str]:
        """Run command on remote host asynchronously.
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            use_env: Whether to use the environment prefix if defined
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd_list = self._build_ssh_command(command, use_env)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
                success = (process.returncode == 0)
                return success, stdout_bytes.decode(), stderr_bytes.decode()
                
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.communicate()
                except ProcessLookupError:
                    pass
                return False, "", f"Command timed out after {timeout} seconds"
                
        except Exception as e:
            return False, "", f"SSH error: {str(e)}"

    def _build_ssh_command(self, command: str, use_env: bool) -> list[str]:
        """Build the raw SSH command list."""
        if use_env and self.env_command:
            # Logic duplicated from synchronous method for consistency
            # If we're using pixi, try to find it in common locations if standard command fails
            if self.env_command.startswith("pixi run") and self.home_dir:
                pixi_locations = [
                    "pixi",  # Try PATH first
                    f"{self.home_dir.rstrip('/')}/.pixi/bin/pixi",
                    "~/.pixi/bin/pixi"
                ]
                # Construct a shell command that tries multiple pixi locations
                pixi_check = " || ".join([f"command -v {loc}" for loc in pixi_locations])
                if self.project_dir:
                    # Change to project directory before running pixi
                    # Handle environment flags if present
                    if "-e" in self.env_command:
                        # Extract environment name from "pixi run -e cuda"
                        parts = self.env_command.split()
                        if "-e" in parts and len(parts) > parts.index("-e") + 1:
                            env = parts[parts.index("-e") + 1]
                            command = f"PIXI_BIN=$({pixi_check} | head -n 1); cd {self.project_dir} && $PIXI_BIN run -e {env} -- {command}"
                        else:
                            command = f"PIXI_BIN=$({pixi_check} | head -n 1); cd {self.project_dir} && $PIXI_BIN run {command}"
                    else:
                        command = f"PIXI_BIN=$({pixi_check} | head -n 1); cd {self.project_dir} && $PIXI_BIN run {command}"
                else:
                    command = f"PIXI_BIN=$({pixi_check} | head -n 1); $PIXI_BIN run {command}"
            else:
                command = f"{self.env_command} {command}"

        # Wrap command in a login shell to ensure PATH and other env vars are loaded
        wrapped_command = f"bash -l -c '{command}'"

        return [
            self.ssh_executable,
            f'-p{self.port}',
            *self.ssh_options,
            f'{self.user}@{self.host}',
            wrapped_command
        ]

    def _find_ssh_executable(self) -> str:
        """Find the SSH executable on the system."""
        # Try finding in PATH first
        ssh_path = shutil.which("ssh")
        if ssh_path:
            return ssh_path
            
        # Fallback to common locations
        common_paths = ["/usr/bin/ssh", "/usr/local/bin/ssh", "/bin/ssh"]
        for path in common_paths:
            if Path(path).exists():
                return path
                
        return "ssh"  # Fallback to name and hope for the best at runtime

    
    def run_command(self, command: str, timeout: int = 30, use_env: bool = True) -> tuple[bool, str, str]:
        """Run command on remote host
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            use_env: Whether to use the environment prefix if defined
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        if use_env and self.env_command:
            # If we're using pixi, try to find it in common locations if standard command fails
            if self.env_command.startswith("pixi run") and self.home_dir:
                pixi_locations = [
                    "pixi",  # Try PATH first
                    f"{self.home_dir.rstrip('/')}/.pixi/bin/pixi",
                    "~/.pixi/bin/pixi"
                ]
                # Construct a shell command that tries multiple pixi locations
                pixi_check = " || ".join([f"command -v {loc}" for loc in pixi_locations])
                if self.project_dir:
                    # Change to project directory before running pixi
                    # Handle environment flags if present
                    if "-e" in self.env_command:
                        # Extract environment name from "pixi run -e cuda"
                        parts = self.env_command.split()
                        if "-e" in parts and len(parts) > parts.index("-e") + 1:
                            env = parts[parts.index("-e") + 1]
                            command = f"PIXI_BIN=$({pixi_check} | head -n 1); cd {self.project_dir} && $PIXI_BIN run -e {env} -- {command}"
                        else:
                            command = f"PIXI_BIN=$({pixi_check} | head -n 1); cd {self.project_dir} && $PIXI_BIN run {command}"
                    else:
                        command = f"PIXI_BIN=$({pixi_check} | head -n 1); cd {self.project_dir} && $PIXI_BIN run {command}"
                else:
                    command = f"PIXI_BIN=$({pixi_check} | head -n 1); $PIXI_BIN run {command}"
            else:
                command = f"{self.env_command} {command}"

        # Wrap command in a login shell to ensure PATH and other env vars are loaded
        # Using bash -l -c handles the case where pixi/conda are defined in .bashrc or .profile
        wrapped_command = f"bash -l -c '{command}'"

        ssh_cmd = [
            self.ssh_executable,
            f'-p{self.port}',
            *self.ssh_options,
            f'{self.user}@{self.host}',
            wrapped_command
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
        """Copy file to remote host"""
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
        except Exception:
            return False

    async def copy_file_async(self, local_path: str, remote_path: str) -> bool:
        """Copy file to remote host asynchronously."""
        scp_cmd = [
            'scp',
            '-P', str(self.port),
            *self.ssh_options,
            local_path,
            f'{self.user}@{self.host}:{remote_path}'
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    async def pull_file_async(self, remote_path: str, local_path: str) -> bool:
        """Pull file from remote host asynchronously."""
        scp_cmd = [
            'scp',
            '-P', str(self.port),
            *self.ssh_options,
            f'{self.user}@{self.host}:{remote_path}',
            local_path
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    async def pull_dir_async(self, remote_path: str, local_path: str, include_pattern: str | None = None) -> bool:
        """Pull directory from remote host asynchronously using rsync."""
        # Ensure local directory exists
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        # Use rsync for efficient directory pulling
        # If remote_path doesn't end in /, rsync creates the directory inside local_path.
        # We ensure it ends in / to sync contents if that's what's intended, 
        # but here we usually want the specifically named directory.
        
        ssh_opts_str = " ".join(self.ssh_options)
        rsync_cmd = [
            'rsync',
            '-avz',
            '-e', f'{self.ssh_executable} -p {self.port} {ssh_opts_str}',
        ]
        
        if include_pattern:
            rsync_cmd.extend(['--include', include_pattern, '--exclude', '*'])
            
        rsync_cmd.extend([
            f'{self.user}@{self.host}:{remote_path}',
            local_path
        ])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
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
        success, stdout, stderr = self.run_command('python3 --version', timeout=5)
        checks['python'] = success
        if success:
            checks['python_version'] = stdout.strip()
        else:
            checks['error'] = f"python3 not found: {stderr.strip()}"
            return False, checks
        
        # Check PyTorch
        success, stdout, stderr = self.run_command('python3 -c "import torch; print(torch.__version__)"', timeout=10)
        checks['pytorch'] = success
        if success:
            checks['pytorch_version'] = stdout.strip()
        else:
            checks['pytorch_error'] = stderr.strip() if stderr else "torch not installed"
        
        # Check CUDA availability
        success, stdout, stderr = self.run_command('python3 -c "import torch; print(torch.cuda.is_available())"', timeout=10)
        checks['cuda'] = success and 'True' in stdout.strip()
        
        return all([checks.get('python', False), checks.get('pytorch', False)]), checks
