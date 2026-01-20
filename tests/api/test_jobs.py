from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from apps.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ugro-api"}

@patch("apps.api.services.job_service.ClusterStateManager")
@patch("apps.api.services.job_service.ResultAggregator")
def test_list_jobs(mock_aggregator_cls, mock_state_manager_cls):
    # Mock State Manager
    mock_state_manager = mock_state_manager_cls.return_value
    mock_state_manager.refresh.return_value.jobs = {
        "job_1": MagicMock(status="running", model="llama-7b", started_at="2023-01-01T10:00:00", gpu_nodes=["node1"]),
        "job_2": MagicMock(status="completed", model="resnet50", started_at="2023-01-01T09:00:00", gpu_nodes=["node2"])
    }
    
    # Mock Result Aggregator
    mock_aggregator = mock_aggregator_cls.return_value
    mock_aggregator.get_job_summary.return_value = {"total_steps": 100, "final_loss": 0.5}

    response = client.get("/api/v1/jobs/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["job_id"] == "job_1" # sorted desc by started_at
    assert data[1]["job_id"] == "job_2"
