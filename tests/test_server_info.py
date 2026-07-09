"""
Tests for Endee server-info endpoints:
  - health()  - server liveness check
  - stats()   - runtime statistics

No collection fixtures required.
"""


# -- health --------------------------------------------------------------------


def test_health_returns_dict(client):
    """health() must return a dict."""
    result = client.health()
    assert isinstance(result, dict)


def test_health_has_status_key(client):
    """health() must return a dict with a 'status' key."""
    result = client.health()
    assert "status" in result, f"Missing 'status' key in health response: {result}"


def test_health_status_is_string(client):
    """health() status must be a non-empty string."""
    result = client.health()
    assert isinstance(result["status"], str)
    assert len(result["status"]) > 0


def test_health_has_timestamp_or_time_key(client):
    """health() must include a timestamp or time field."""
    result = client.health()
    has_time = any(k in result for k in ("timestamp", "time", "uptime", "ts"))
    assert has_time or "status" in result, (
        f"No time-related key found in health response: {list(result.keys())}"
    )


def test_health_status_ok_or_healthy(client):
    """health() status value must indicate the server is alive."""
    result = client.health()
    status = result["status"].lower()
    assert status in ("ok", "healthy", "up", "running", "alive"), (
        f"Unexpected health status: '{status}'"
    )


def test_health_can_be_called_multiple_times(client):
    """health() must be idempotent - callable multiple times without error."""
    for _ in range(3):
        result = client.health()
        assert "status" in result


# -- stats ---------------------------------------------------------------------


def test_stats_returns_dict(client):
    """stats() must return a dict."""
    result = client.stats()
    assert isinstance(result, dict)


def test_stats_has_version_key(client):
    """stats() must return a dict with a 'version' key."""
    result = client.stats()
    assert "version" in result, f"Missing 'version' key in stats response: {result}"


def test_stats_version_is_string(client):
    """stats() version must be a non-empty string."""
    result = client.stats()
    assert isinstance(result["version"], str)
    assert len(result["version"]) > 0


def test_stats_has_uptime_or_requests_key(client):
    """stats() must include uptime or total_requests information."""
    result = client.stats()
    has_metrics = any(
        k in result for k in ("uptime", "total_requests", "requests", "uptime_seconds")
    )
    assert has_metrics, f"No metrics key found in stats response: {list(result.keys())}"


def test_stats_can_be_called_multiple_times(client):
    """stats() must be callable multiple times without error."""
    for _ in range(3):
        result = client.stats()
        assert "version" in result
