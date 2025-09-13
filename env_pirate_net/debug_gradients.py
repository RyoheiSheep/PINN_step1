#!/usr/bin/env python3
"""
Debug gradient flow issues
"""

import torch
from cavity_flow import CavityFlowProblem
from pirate_network import create_network
from settings import Config

def debug_gradient_flow():
    config = Config()
    physics = CavityFlowProblem(config.physics)
    network = create_network(config.network)
    
    print("Network parameters:")
    for name, param in network.named_parameters():
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    print("\nNetwork buffers:")
    for name, buffer in network.named_buffers():
        print(f"  {name}: {buffer.shape}")
    
    # Test with simple loss
    print("\n--- Testing simple forward pass ---")
    colloc_pts = physics.generate_collocation_points(10)
    
    # Test boundary loss only
    boundary_losses = physics.compute_boundary_loss(network)
    boundary_loss = boundary_losses["boundary_total"]
    
    print(f"Boundary loss: {boundary_loss.item()}")
    boundary_loss.backward()
    
    params_with_grad = 0
    for name, param in network.named_parameters():
        has_grad = param.grad is not None
        if has_grad:
            params_with_grad += 1
        print(f"  {name}: grad = {'✓' if has_grad else '✗'}")
    
    print(f"\nBoundary loss gradients: {params_with_grad}/{len(list(network.parameters()))}")
    
    # Clear gradients
    network.zero_grad()
    
    # Test physics loss only
    print("\n--- Testing physics loss ---")
    physics_losses = physics.compute_physics_loss(network, colloc_pts)
    physics_loss = physics_losses["physics_total"]
    
    print(f"Physics loss: {physics_loss.item()}")
    physics_loss.backward()
    
    params_with_grad = 0
    for name, param in network.named_parameters():
        has_grad = param.grad is not None
        if has_grad:
            params_with_grad += 1
        print(f"  {name}: grad = {'✓' if has_grad else '✗'}")
    
    print(f"\nPhysics loss gradients: {params_with_grad}/{len(list(network.parameters()))}")

if __name__ == "__main__":
    debug_gradient_flow()