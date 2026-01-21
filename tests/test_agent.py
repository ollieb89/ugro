# tests/test_agent.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_ugro_agent_creation():
    """Test UGROAgent can be created"""
    from ugro.agent import UGROAgent
    
    # Test with default config
    agent = UGROAgent()
    assert agent is not None
    assert hasattr(agent, 'logger')
    
    # Test with custom config
    test_config = {
        'cluster': {
            'name': 'Test Cluster',
            'master': {'hostname': 'test-master', 'ip': '192.168.1.100'},
            'communication': {'master_port': 29500}
        },
        'workers': []
    }
    
    agent = UGROAgent(config=test_config)
    assert agent is not None

def test_ugro_agent_validate_cluster():
    """Test cluster validation"""
    from ugro.agent import UGROAgent
    
    agent = UGROAgent()
    
    # Mock the cluster check method
    with patch.object(agent, 'check_cluster_connectivity', return_value=True):
        result = agent.validate_cluster()
        assert result is True
