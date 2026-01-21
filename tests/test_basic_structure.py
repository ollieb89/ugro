# tests/test_basic_structure.py
def test_project_structure():
    """Test that all required directories and files exist"""
    from pathlib import Path
    
    # Check directories exist
    assert Path("src/ugro").exists()
    assert Path("config").exists()
    assert Path("data").exists()
    assert Path("logs").exists()
    assert Path("tests").exists()
    
    # Check main module file exists
    assert Path("src/ugro/__init__.py").exists()
    
    # Check version in __init__.py
    init_content = Path("src/ugro/__init__.py").read_text()
    assert '__version__ = "0.1.0"' in init_content
