import re
from typing import Optional

def validate_wandb_api_key(api_key: Optional[str]) -> bool:
    """Validate W&B API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    # W&B API keys are typically 40 characters long and alphanumeric
    return bool(re.match(r'^[a-zA-Z0-9_]{20,}$', api_key))

def mask_api_key(api_key: str) -> str:
    """Mask API key for logging, showing first and last 4 chars."""
    if not api_key or len(api_key) < 8:
        return "***"
    return f"{api_key[:12]}{'*' * (len(api_key) - 16)}{api_key[-4:]}"

def validate_project_name(project_name: Optional[str]) -> bool:
    """Validate W&B project name."""
    if not project_name or not isinstance(project_name, str):
        return False
    # Project names should be alphanumeric with hyphens/underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', project_name)) and 1 <= len(project_name) <= 128
