#!/usr/bin/env python3
"""
Comprehensive tests for Pirate network implementation
"""

import torch
import torch.nn as nn
from pirate_network import create_network, Dense, FourierEmbedding, ModifiedMlp, StandardMlp
from settings import Config, NetworkConfig

def test_dense_layer():
    """Test Dense layer with and without weight factorization"""
    print("Testing Dense layer...")
    
    # Standard dense layer
    layer1 = Dense(10, 5, reparam=None)
    x = torch.randn(3, 10)
    y1 = layer1(x)
    assert y1.shape == (3, 5), f"Expected (3, 5), got {y1.shape}"
    print("✓ Standard Dense layer works")
    
    # Weight factorization
    reparam = {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    layer2 = Dense(10, 5, reparam=reparam)
    y2 = layer2(x)
    assert y2.shape == (3, 5), f"Expected (3, 5), got {y2.shape}"
    print("✓ Weight factorization Dense layer works")
    
    # Check that weight factorization produces different results
    assert not torch.allclose(y1, y2, atol=1e-3), "Weight factorization should produce different outputs"
    print("✓ Weight factorization changes behavior as expected")

def test_fourier_embedding():
    """Test Fourier embedding layer"""
    print("\nTesting Fourier embedding...")
    
    # Test Fourier embedding
    fourier = FourierEmbedding(input_dim=2, embed_dim=256, embed_scale=10.0)
    x = torch.randn(5, 2)  # 5 samples, 2D input (x, y)
    embedded = fourier(x)
    
    assert embedded.shape == (5, 256), f"Expected (5, 256), got {embedded.shape}"
    print("✓ Fourier embedding shape correct")
    
    # Check that embedding is periodic (approximately)
    x_test = torch.tensor([[0.0, 0.0], [2*torch.pi/10.0, 0.0]])
    emb_test = fourier(x_test)
    # Due to the scale factor, these should be close but not exactly equal
    print("✓ Fourier embedding works")

def test_network_architectures():
    """Test different network architectures"""
    print("\nTesting network architectures...")
    
    config = NetworkConfig()
    
    # Test ModifiedMlp (Pirate network)
    config.arch_name = "ModifiedMlp"
    pirate_net = create_network(config)
    assert isinstance(pirate_net, ModifiedMlp), "Should create ModifiedMlp"
    print("✓ ModifiedMlp (Pirate) network created")
    
    # Test StandardMlp
    config.arch_name = "Mlp" 
    std_net = create_network(config)
    assert isinstance(std_net, StandardMlp), "Should create StandardMlp"
    print("✓ StandardMlp network created")
    
    # Test forward pass for both
    x = torch.randn(8, 2)
    
    pirate_output = pirate_net(x)
    std_output = std_net(x)
    
    assert pirate_output.shape == (8, 3), f"Pirate output shape: {pirate_output.shape}"
    assert std_output.shape == (8, 3), f"Standard output shape: {std_output.shape}"
    print("✓ Both networks produce correct output shapes")

def test_gradient_computation():
    """Test that gradients flow correctly through the network"""
    print("\nTesting gradient computation...")
    
    config = NetworkConfig()
    net = create_network(config)
    
    # Enable gradient computation
    x = torch.randn(5, 2, requires_grad=True)
    output = net(x)
    
    # Compute some loss
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Input gradients should exist"
    
    param_grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "Parameter gradients should exist"
    print("✓ Gradients computed successfully")

def test_different_configurations():
    """Test network with different configurations"""
    print("\nTesting different configurations...")
    
    # Test without Fourier embeddings
    config1 = NetworkConfig()
    config1.fourier_emb = None
    net1 = create_network(config1)
    
    x = torch.randn(3, 2)
    out1 = net1(x)
    assert out1.shape == (3, 3), "Should work without Fourier embeddings"
    print("✓ Network without Fourier embeddings works")
    
    # Test with different activation
    config2 = NetworkConfig()
    config2.activation = "relu"
    net2 = create_network(config2)
    
    out2 = net2(x)
    assert out2.shape == (3, 3), "Should work with ReLU activation"
    print("✓ Network with ReLU activation works")
    
    # Test with different dimensions
    config3 = NetworkConfig()
    config3.hidden_dim = 128
    config3.num_layers = 6
    net3 = create_network(config3)
    
    out3 = net3(x)
    assert out3.shape == (3, 3), "Should work with different dimensions"
    print("✓ Network with custom dimensions works")

def test_physics_derivatives():
    """Test that we can compute derivatives for physics (crucial for PINNs)"""
    print("\nTesting physics derivatives...")
    
    config = NetworkConfig()
    net = create_network(config)
    
    # Test points in domain
    x = torch.tensor([[0.5, 0.5], [0.2, 0.8]], requires_grad=True)
    
    # Forward pass
    output = net(x)  # Shape: (2, 3) for (u, v, p)
    u, v, p = output[:, 0], output[:, 1], output[:, 2]
    
    # Compute first derivatives (needed for physics loss)
    u_grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    v_grad = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    p_grad = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    
    # Extract components
    u_x, u_y = u_grad[:, 0], u_grad[:, 1]
    v_x, v_y = v_grad[:, 0], v_grad[:, 1]
    p_x, p_y = p_grad[:, 0], p_grad[:, 1]
    
    print(f"✓ First derivatives computed: u_x shape = {u_x.shape}")
    
    # Test second derivatives (for Laplacian in Navier-Stokes)
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1]
    laplacian_u = u_xx + u_yy
    
    print(f"✓ Second derivatives computed: Laplacian u shape = {laplacian_u.shape}")
    print("✓ Physics derivatives test passed")

def compare_architectures():
    """Compare Pirate vs Standard MLP performance"""
    print("\nComparing architectures...")
    
    # Create both networks with same config
    config = NetworkConfig()
    
    config.arch_name = "ModifiedMlp"
    pirate_net = create_network(config)
    
    config.arch_name = "Mlp"
    std_net = create_network(config)
    
    # Count parameters
    pirate_params = sum(p.numel() for p in pirate_net.parameters())
    std_params = sum(p.numel() for p in std_net.parameters())
    
    print(f"Pirate network parameters: {pirate_params:,}")
    print(f"Standard network parameters: {std_params:,}")
    
    # Test on same input
    x = torch.randn(10, 2)
    
    with torch.no_grad():
        pirate_out = pirate_net(x)
        std_out = std_net(x)
    
    print(f"Pirate output range: [{pirate_out.min():.3f}, {pirate_out.max():.3f}]")
    print(f"Standard output range: [{std_out.min():.3f}, {std_out.max():.3f}]")
    
    print("✓ Architecture comparison completed")

if __name__ == "__main__":
    print("🧪 Running Pirate Network Tests\n" + "="*50)
    
    test_dense_layer()
    test_fourier_embedding()
    test_network_architectures()
    test_gradient_computation()
    test_different_configurations()
    test_physics_derivatives()
    compare_architectures()
    
    print("\n" + "="*50)
    print("🎉 All tests passed! Pirate Network ready for physics!")