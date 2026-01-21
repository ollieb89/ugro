# tests/test_config.py
def test_load_cluster_config():
    """Test loading cluster configuration from YAML"""
    from src.ugro.utils import load_cluster_config
    
    config = load_cluster_config()
    
    assert 'cluster' in config
    assert 'workers' in config
    assert config['cluster']['name'] == "Home AI Lab"
    assert len(config['workers']) == 2
    assert config['workers'][0]['name'] == 'gpu1'
    assert config['workers'][1]['name'] == 'gpu2'
