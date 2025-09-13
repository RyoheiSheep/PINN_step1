#!/usr/bin/env python3
"""
Comprehensive tests for cavity flow physics implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from cavity_flow import CavityFlowProblem
from pirate_network import create_network
from settings import Config, PhysicsConfig

def test_point_generation():
    """Test collocation and boundary point generation"""
    print("Testing point generation...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics)
    
    # Test collocation points
    n_colloc = 500
    colloc_pts = physics.generate_collocation_points(n_colloc, random=True)
    assert colloc_pts.shape == (n_colloc, 2), f"Expected {(n_colloc, 2)}, got {colloc_pts.shape}"
    
    # Check bounds
    x, y = colloc_pts[:, 0], colloc_pts[:, 1]
    assert torch.all(x >= 0) and torch.all(x <= 1), "X coordinates out of bounds"
    assert torch.all(y >= 0) and torch.all(y <= 1), "Y coordinates out of bounds"
    print("✓ Collocation points in bounds")
    
    # Test regular grid
    grid_pts = physics.generate_collocation_points(100, random=False)
    assert grid_pts.shape[0] <= 100, "Grid points exceed requested"
    print("✓ Regular grid generation works")
    
    # Test boundary points
    boundary_pts = physics.boundary_points
    assert boundary_pts.shape[1] == 2, "Boundary points should be 2D"
    print(f"✓ Boundary points shape: {boundary_pts.shape}")
    
    # Check boundary conditions
    assert len(physics.u_bc) == len(boundary_pts), "u_bc size mismatch"
    assert len(physics.v_bc) == len(boundary_pts), "v_bc size mismatch"
    
    # Top wall should have u=1, others u=0
    num_per_side = len(boundary_pts) // 4
    assert torch.allclose(physics.u_bc[:num_per_side], torch.ones(num_per_side)), "Top wall u_bc incorrect"
    assert torch.allclose(physics.u_bc[num_per_side:], torch.zeros(3*num_per_side)), "Other walls u_bc incorrect"
    assert torch.allclose(physics.v_bc, torch.zeros_like(physics.v_bc)), "All walls should have v=0"
    print("✓ Boundary conditions correct")

def test_physics_derivatives():
    """Test that physics derivatives are computed correctly"""
    print("\nTesting physics derivatives...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics)
    network = create_network(config.network)
    
    # Test points
    test_pts = torch.tensor([[0.5, 0.5], [0.2, 0.8]], requires_grad=True)
    
    # Compute residuals
    ru, rv, rc = physics.compute_physics_residuals(network, test_pts)
    
    assert ru.shape == (2,), f"ru shape: {ru.shape}"
    assert rv.shape == (2,), f"rv shape: {rv.shape}"  
    assert rc.shape == (2,), f"rc shape: {rc.shape}"
    print("✓ Residual computation shapes correct")
    
    # Check that residuals change with input
    test_pts2 = torch.tensor([[0.3, 0.3], [0.7, 0.7]], requires_grad=True)
    ru2, rv2, rc2 = physics.compute_physics_residuals(network, test_pts2)
    
    assert not torch.allclose(ru, ru2, atol=1e-3), "Residuals should change with input"
    assert not torch.allclose(rv, rv2, atol=1e-3), "Residuals should change with input"
    assert not torch.allclose(rc, rc2, atol=1e-3), "Residuals should change with input"
    print("✓ Residuals vary with input coordinates")

def test_loss_computation():
    """Test loss function computation"""
    print("\nTesting loss computation...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics)
    network = create_network(config.network)
    
    # Generate points
    colloc_pts = physics.generate_collocation_points(200)
    
    # Test boundary loss
    boundary_losses = physics.compute_boundary_loss(network)
    required_keys = ["u_bc", "v_bc", "boundary_total"]
    for key in required_keys:
        assert key in boundary_losses, f"Missing boundary loss key: {key}"
        assert isinstance(boundary_losses[key], torch.Tensor), f"{key} should be tensor"
        assert boundary_losses[key].numel() == 1, f"{key} should be scalar"
    print("✓ Boundary loss computation")
    
    # Test physics loss
    physics_losses = physics.compute_physics_loss(network, colloc_pts)
    required_keys = ["ru", "rv", "rc", "physics_total"]
    for key in required_keys:
        assert key in physics_losses, f"Missing physics loss key: {key}"
        assert isinstance(physics_losses[key], torch.Tensor), f"{key} should be tensor"
        assert physics_losses[key].numel() == 1, f"{key} should be scalar"
    print("✓ Physics loss computation")
    
    # Test total loss
    total_loss, loss_dict = physics.compute_total_loss(network, colloc_pts)
    assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
    assert total_loss.numel() == 1, "Total loss should be scalar"
    assert "total" in loss_dict, "Loss dict should contain 'total'"
    assert torch.isclose(total_loss, loss_dict["total"]), "Total loss inconsistency"
    print("✓ Total loss computation")

def test_different_reynolds_numbers():
    """Test physics with different Reynolds numbers"""
    print("\nTesting different Reynolds numbers...")
    
    re_values = [10, 100, 1000]
    
    for re in re_values:
        config = Config()
        config.physics.reynolds_number = re
        config._setup_derived_params()  # Recalculate viscosity
        
        physics = CavityFlowProblem(config.physics)
        network = create_network(config.network)
        
        colloc_pts = physics.generate_collocation_points(100)
        total_loss, _ = physics.compute_total_loss(network, colloc_pts)
        
        print(f"  Re={re}: ν={physics.nu:.6f}, loss={total_loss.item():.2f}")
    
    print("✓ Different Reynolds numbers work")

def test_solution_evaluation():
    """Test solution evaluation and visualization setup"""
    print("\nTesting solution evaluation...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics)
    network = create_network(config.network)
    
    # Evaluate solution on grid
    solution = physics.evaluate_solution(network, grid_resolution=20)
    
    required_fields = ["x", "y", "u", "v", "p", "speed"]
    for field in required_fields:
        assert field in solution, f"Missing solution field: {field}"
        assert isinstance(solution[field], np.ndarray), f"{field} should be numpy array"
    
    # Check shapes
    expected_shape = (20, 20)
    for field in ["u", "v", "p", "speed"]:
        assert solution[field].shape == expected_shape, f"{field} shape: {solution[field].shape}"
    
    print("✓ Solution evaluation works")
    print(f"  Grid shape: {solution['u'].shape}")
    print(f"  U range: [{solution['u'].min():.3f}, {solution['u'].max():.3f}]")
    print(f"  V range: [{solution['v'].min():.3f}, {solution['v'].max():.3f}]")
    print(f"  Speed range: [{solution['speed'].min():.3f}, {solution['speed'].max():.3f}]")

def visualize_problem_setup():
    """Visualize the problem setup (points, boundaries, etc.)"""
    print("\nCreating problem setup visualization...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics)
    
    # Generate points
    colloc_pts = physics.generate_collocation_points(500, random=True)
    boundary_pts = physics.boundary_points
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot point distribution
    axes[0].scatter(colloc_pts[:, 0], colloc_pts[:, 1], c='blue', s=1, alpha=0.6, label='Collocation')
    axes[0].scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='red', s=2, label='Boundary')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Training Points Distribution')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot boundary conditions
    num_per_side = len(boundary_pts) // 4
    
    # Top wall (moving lid)
    top_pts = boundary_pts[:num_per_side]
    axes[1].quiver(top_pts[:, 0], top_pts[:, 1], 
                   physics.u_bc[:num_per_side], physics.v_bc[:num_per_side], 
                   color='red', scale=5, width=0.003, label='Moving lid (u=1)')
    
    # Other walls (no-slip)
    other_pts = boundary_pts[num_per_side:]
    axes[1].scatter(other_pts[:, 0], other_pts[:, 1], c='blue', s=10, label='No-slip walls')
    
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Boundary Conditions')
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cavity_flow_setup.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Problem setup visualization saved as 'cavity_flow_setup.png'")

def test_gradient_flow():
    """Test that gradients flow through physics loss"""
    print("\nTesting gradient flow...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics)
    network = create_network(config.network)
    
    # Generate points and compute loss
    colloc_pts = physics.generate_collocation_points(50)
    total_loss, _ = physics.compute_total_loss(network, colloc_pts)
    
    # Compute gradients
    total_loss.backward()
    
    # Check that network parameters have gradients
    param_with_grads = [p for p in network.parameters() if p.grad is not None]
    total_params = list(network.parameters())
    
    print(f"  Parameters with gradients: {len(param_with_grads)}/{len(total_params)}")
    
    # Check that most parameters have gradients (some might not be used)
    # Note: Fourier embedding weights are buffers (non-trainable), so exclude them
    trainable_params = [p for p in network.parameters() if p.requires_grad]
    grad_ratio = len(param_with_grads) / len(trainable_params)
    
    print(f"  Trainable parameters: {len(trainable_params)}")
    assert grad_ratio > 0.95, f"Too few trainable parameters have gradients: {grad_ratio:.2%}"
    
    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in param_with_grads]
    avg_grad_norm = np.mean(grad_norms)
    max_grad_norm = max(grad_norms) if grad_norms else 0
    
    assert avg_grad_norm > 1e-8, "Gradients too small"
    
    print(f"✓ Gradients computed for {len(param_with_grads)} parameters")
    print(f"  Average gradient norm: {avg_grad_norm:.6f}")
    print(f"  Max gradient norm: {max_grad_norm:.6f}")

if __name__ == "__main__":
    print("🧪 Running Cavity Flow Physics Tests")
    print("=" * 50)
    
    test_point_generation()
    test_physics_derivatives()
    test_loss_computation()
    test_different_reynolds_numbers()
    test_solution_evaluation()
    visualize_problem_setup()
    test_gradient_flow()
    
    print("\n" + "=" * 50)
    print("🎉 All physics tests passed! Ready for training!")