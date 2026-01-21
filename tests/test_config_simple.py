# tests/test_config_simple.py
import yaml
from pathlib import Path

def test_load_cluster_config_simple():
    """Test loading cluster configuration from YAML directly"""
    config_path = Path("config/cluster.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    assert 'cluster' in config
    assert 'workers' in config
    assert config['cluster']['name'] == "Home AI Lab"
    assert len(config['workers']) == 2
    assert config['workers'][0]['name'] == 'gpu1'
    assert config['workers'][1]['name'] == 'gpu2'
