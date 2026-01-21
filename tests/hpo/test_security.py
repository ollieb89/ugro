# tests/hpo/test_security.py
import pytest
from ugro.hpo.security import validate_wandb_api_key, mask_api_key, validate_project_name

def test_validate_wandb_api_key_valid():
    # Valid W&B API key format
    valid_key = "wandb_api_key_1234567890abcdef"
    assert validate_wandb_api_key(valid_key) == True

def test_validate_wandb_api_key_invalid():
    # Invalid formats
    assert validate_wandb_api_key("") == False
    assert validate_wandb_api_key("short") == False
    assert validate_wandb_api_key(None) == False

def test_mask_api_key():
    key = "wandb_api_key_1234567890abcdef"
    masked = mask_api_key(key)
    # First 12 chars: wandb_api_ke
    # Last 4 chars: cdef
    # Middle 14 chars masked: **************
    assert masked == "wandb_api_ke**************cdef"
    assert len(masked) == len(key)

def test_validate_project_name():
    assert validate_project_name("valid-project") == True
    assert validate_project_name("invalid project!") == False
    assert validate_project_name("") == False
