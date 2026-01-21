#!/usr/bin/env python3
"""Simple test runner for W&B security functions."""

import sys
from pathlib import Path

# Add src/ugro to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "ugro"))

# Direct import to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location("security", Path(__file__).parent / "src" / "ugro" / "hpo" / "security.py")
security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(security)

validate_wandb_api_key = security.validate_wandb_api_key
mask_api_key = security.mask_api_key
validate_project_name = security.validate_project_name

def test_api_key_validation():
    """Test API key validation."""
    print("Testing API key validation...")
    
    # Valid keys
    valid_keys = [
        "wandb_api_key_1234567890abcdef",
        "0123456789abcdef0123456789abcdef01234567",
        "test_key_1234567890abcdef"
    ]
    
    for key in valid_keys:
        assert validate_wandb_api_key(key) == True, f"Key {key} should be valid"
    print("✓ Valid keys accepted")
    
    # Invalid keys
    invalid_keys = [
        "",
        "short",
        None,
        "invalid key!",
        "123",
        "a" * 19  # Too short
    ]
    
    for key in invalid_keys:
        assert validate_wandb_api_key(key) == False, f"Key {key} should be invalid"
    print("✓ Invalid keys rejected")

def test_api_key_masking():
    """Test API key masking."""
    print("\nTesting API key masking...")
    
    # Normal key (40 chars)
    key = "0123456789abcdef0123456789abcdef01234567"
    masked = mask_api_key(key)
    # First 12 + 24 stars + last 4
    expected = "0123456789ab************************4567"
    print(f"Expected: {expected}")
    print(f"Got:      {masked}")
    assert masked == expected, f"Expected {expected}, got {masked}"
    assert len(masked) == len(key)
    print(f"✓ Key masked correctly")
    
    # Short key
    short_key = "1234"
    masked_short = mask_api_key(short_key)
    assert masked_short == "***"
    print(f"✓ Short key masked: {masked_short}")

def test_project_name_validation():
    """Test project name validation."""
    print("\nTesting project name validation...")
    
    # Valid names
    valid_names = [
        "valid-project",
        "valid_project",
        "project123",
        "a",  # Minimum length
        "a" * 128  # Maximum length
    ]
    
    for name in valid_names:
        assert validate_project_name(name) == True, f"Name {name} should be valid"
    print("✓ Valid project names accepted")
    
    # Invalid names
    invalid_names = [
        "",
        None,
        "invalid project!",
        "project@name",
        "project.name",
        "a" * 129  # Too long
    ]
    
    for name in invalid_names:
        assert validate_project_name(name) == False, f"Name {name} should be invalid"
    print("✓ Invalid project names rejected")

def main():
    """Run all security tests."""
    print("🔒 Running W&B Security Tests\n")
    
    try:
        test_api_key_validation()
        test_api_key_masking()
        test_project_name_validation()
        
        print("\n✅ All security tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
