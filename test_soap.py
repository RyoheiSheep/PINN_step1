#!/usr/bin/env python3
"""
Test SOAP optimizer with a simple neural network
"""

import sys
sys.path.append('./SOAP')

import torch
import torch.nn as nn
from soap import SOAP

# Simple test network
class SimpleNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

def test_soap_optimizer():
    print("Testing SOAP optimizer...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple network
    net = SimpleNet().to(device)
    
    # Create SOAP optimizer
    optimizer = SOAP(
        net.parameters(),
        lr=3e-3,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10
    )
    
    print("SOAP optimizer created successfully!")
    
    # Test with dummy data
    batch_size = 100
    x = torch.randn(batch_size, 2, device=device, requires_grad=True)
    target = torch.randn(batch_size, 3, device=device)
    
    # Simple training loop test
    for step in range(5):
        optimizer.zero_grad()
        
        output = net(x)
        loss = nn.MSELoss()(output, target)
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.6f}")
    
    print("✓ SOAP optimizer test completed successfully!")
    return True

if __name__ == "__main__":
    test_soap_optimizer()