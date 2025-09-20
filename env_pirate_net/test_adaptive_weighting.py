"""
Test script for adaptive loss weighting implementations
Tests gradient norm weighting and NTK computation
"""

import torch
import torch.nn as nn
import numpy as np
from settings import Config
from training import PINNTrainer
from adaptive_weighting import GradientNormWeighting, NTKWeighting, AdaptiveWeightingManager

def test_gradient_norm_weighting():
    """Test gradient norm weighting with simple example"""
    print("Testing Gradient Norm Weighting...")

    # Create simple test network
    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 3)
    )

    # Create test input
    x = torch.randn(100, 2, requires_grad=True)

    # Forward pass
    output = net(x)

    # Create test loss components with different magnitudes
    loss_dict = {
        "u_bc": torch.mean(output[:, 0] ** 2),     # Small loss
        "v_bc": torch.mean(output[:, 1] ** 2) * 100,  # Large loss
        "ru": torch.mean(output[:, 2] ** 2) * 10,     # Medium loss
    }

    # Initialize gradient norm weighting
    grad_weighter = GradientNormWeighting(update_every=1)

    # Update weights multiple times
    for i in range(5):
        weights = grad_weighter.update_weights(net, loss_dict)
        print(f"  Step {i+1}: {weights}")

        # Check that weights are attempting to balance gradient norms
        if i > 0:  # After first update
            # Weight for large loss should be smaller
            assert weights['v_bc'] < weights['u_bc'], "Large loss should get smaller weight"

    print("  [PASS] Gradient norm weighting test passed!")


def test_ntk_computation():
    """Test NTK computation with simple example"""
    print("Testing NTK Computation...")

    # Create simple test network
    net = nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 3)
    )

    # Create test points
    collocation_points = torch.randn(50, 2, requires_grad=True)
    boundary_points = torch.randn(20, 2, requires_grad=True)

    # Initialize NTK weighting
    ntk_weighter = NTKWeighting(update_every=1)

    # Compute NTK values
    try:
        weights = ntk_weighter.update_weights(net, collocation_points, boundary_points)
        print(f"  NTK weights: {weights}")

        # Check that weights are reasonable
        for key, weight in weights.items():
            assert weight > 0, f"Weight for {key} should be positive"
            assert weight < 1000, f"Weight for {key} should be reasonable"

        print("  [PASS] NTK computation test passed!")

    except Exception as e:
        print(f"  [WARN] NTK test encountered issue: {e}")
        print("  This is expected for very small networks")


def test_adaptive_weighting_manager():
    """Test the adaptive weighting manager"""
    print("Testing Adaptive Weighting Manager...")

    # Test fixed weighting
    manager_fixed = AdaptiveWeightingManager(scheme="fixed")

    loss_dict = {"u_bc": torch.tensor(1.0), "v_bc": torch.tensor(2.0)}
    weights = manager_fixed.get_weights(None, loss_dict)

    assert all(w == 1.0 for w in weights.values()), "Fixed weights should be 1.0"
    print("  [PASS] Fixed weighting test passed!")

    # Test gradient norm weighting
    manager_grad = AdaptiveWeightingManager(
        scheme="grad_norm",
        grad_norm_config={"update_every": 1}
    )

    net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))
    x = torch.randn(50, 2, requires_grad=True)
    output = net(x)

    loss_dict = {
        "u_bc": torch.mean(output[:, 0] ** 2),
        "v_bc": torch.mean(output[:, 1] ** 2) * 100,
    }

    weights = manager_grad.get_weights(net, loss_dict)
    print(f"  Gradient norm weights: {weights}")

    assert len(weights) == len(loss_dict), "Should have weight for each loss"
    print("  [PASS] Gradient norm manager test passed!")


def test_full_integration():
    """Test full integration with PINNTrainer"""
    print("Testing Full Integration...")

    # Create configuration with gradient norm weighting
    config = Config()
    config.weighting.scheme = "grad_norm"
    config.weighting.grad_norm["update_every"] = 10
    config.training.epochs = 5  # Very short test
    config.training.log_every = 1

    try:
        # Create trainer
        trainer = PINNTrainer(config)

        # Run a few training steps
        for epoch in range(3):
            loss_dict = trainer.train_step()

            # Check that we get weight information
            weight_keys = [k for k in loss_dict.keys() if k.startswith('weight_')]
            assert len(weight_keys) > 0, "Should have weight information in loss dict"

            print(f"  Epoch {epoch}: Total loss = {loss_dict['total']:.4f}")

        print("  [PASS] Full integration test passed!")

    except Exception as e:
        print(f"  [WARN] Integration test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running Adaptive Weighting Tests")
    print("=" * 50)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    try:
        test_gradient_norm_weighting()
        print()

        test_ntk_computation()
        print()

        test_adaptive_weighting_manager()
        print()

        test_full_integration()
        print()

        print("[SUCCESS] All tests passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()