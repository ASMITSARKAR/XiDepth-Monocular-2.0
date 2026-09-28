import pytest
import torch

from utils.health import DisparityHealthMonitor


def test_health_monitor_healthy_initialization():
    # Freshly initialized model has disp ~ 0.011, disp_std ~ 0.0001, depth ~ 8.5m
    monitor = DisparityHealthMonitor(warmup_steps=10)
    disp = torch.full((2, 1, 192, 640), 0.011) + 0.0001 * torch.randn(2, 1, 192, 640)
    disp = torch.clamp(disp, 0.001, 0.999)

    for _ in range(5):
        metrics = monitor.check(disp)

    assert "disp_mean" in metrics
    assert "depth_mean" in metrics
    assert 7.0 <= metrics["depth_mean"] <= 11.0
    assert metrics["disp_std"] > 0


def test_health_monitor_nan_detection():
    monitor = DisparityHealthMonitor()
    disp = torch.full((2, 1, 192, 640), float("nan"))

    with pytest.raises(RuntimeError, match="Numerical instability detected"):
        monitor.check(disp)


def test_health_monitor_inf_detection():
    monitor = DisparityHealthMonitor()
    disp = torch.full((2, 1, 192, 640), float("inf"))

    with pytest.raises(RuntimeError, match="Numerical instability detected"):
        monitor.check(disp)


def test_health_monitor_post_warmup_dead_neuron():
    # If output is completely flat (machine zero variance) after warmup with fail_on_collapse=True
    monitor = DisparityHealthMonitor(
        warmup_steps=3,
        dead_neuron_std_thresh=1e-6,
        max_consecutive_collapse=2,
        fail_on_collapse=True,
    )
    flat_disp = torch.full((2, 1, 192, 640), 0.011)  # Exact identical float across all pixels

    # Warmup steps (steps 1, 2, 3) - no error
    for _ in range(3):
        monitor.check(flat_disp)

    # Step 4: first collapsed step after warmup (consecutive = 1 < 2)
    monitor.check(flat_disp)

    # Step 5: second collapsed step after warmup (consecutive = 2 >= 2) -> raises RuntimeError
    with pytest.raises(RuntimeError, match="Sustained collapse detected: Zero spatial variance"):
        monitor.check(flat_disp)


def test_health_monitor_near_plane_collapse():
    monitor = DisparityHealthMonitor(
        warmup_steps=2,
        dead_neuron_std_thresh=1e-6,
        max_consecutive_collapse=2,
        fail_on_collapse=True,
    )
    # Disparity with non-zero std (to avoid dead neuron check) but saturating near plane (depth < 0.3m)
    # disp near 1.0 -> depth near 0.1m
    near_disp = torch.full((2, 1, 192, 640), 0.99) + 0.005 * torch.randn(2, 1, 192, 640)
    near_disp = torch.clamp(near_disp, 0.95, 0.999)

    for _ in range(2):
        monitor.check(near_disp)

    monitor.check(near_disp)
    with pytest.raises(RuntimeError, match="Near-plane saturation"):
        monitor.check(near_disp)


def test_health_monitor_far_plane_collapse():
    monitor = DisparityHealthMonitor(
        warmup_steps=2,
        dead_neuron_std_thresh=1e-6,
        max_consecutive_collapse=2,
        fail_on_collapse=True,
    )
    # Disparity with non-zero std but saturating far plane (depth > 95.0m)
    # disp near 1e-5 -> scaled_disp near 0.0101 -> depth near 99m (> 95m)
    far_disp = torch.full((2, 1, 192, 640), 1e-5) + 2e-6 * torch.randn(2, 1, 192, 640)
    far_disp = torch.clamp(far_disp, 1e-6, 3e-5)

    for _ in range(2):
        monitor.check(far_disp)

    monitor.check(far_disp)
    with pytest.raises(RuntimeError, match="Far-plane saturation"):
        monitor.check(far_disp)


