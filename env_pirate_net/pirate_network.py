"""
Pirate Network implementation in PyTorch
Converted from JAX/Flax ModifiedMlp architecture

Based on: PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks
Reference: pirate_net/jaxpi/archs.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

from settings import NetworkConfig

class Dense(nn.Module):
    """
    Dense layer with optional weight factorization (reparam)
    Converts JAX flax.linen.Dense to PyTorch equivalent
    """
    
    def __init__(self, in_features: int, out_features: int, reparam: Optional[Dict] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reparam = reparam
        
        if reparam is None:
            # Standard linear layer
            self.linear = nn.Linear(in_features, out_features)
        elif reparam["type"] == "weight_fact":
            # Weight factorization: kernel = g * v
            self.mean = reparam["mean"]
            self.stddev = reparam["stddev"]
            
            # v: base weights initialized with Xavier/Glorot normal
            self.v = nn.Parameter(torch.randn(in_features, out_features))
            nn.init.xavier_normal_(self.v)
            
            # g: multiplicative factors per output feature  
            # Initialized as g = exp(mean + stddev * normal())
            log_g_init = self.mean + self.stddev * torch.randn(out_features)
            self.log_g = nn.Parameter(log_g_init)
            
            # Bias
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            raise ValueError(f"Unsupported reparam type: {reparam['type']}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam is None:
            return self.linear(x)
        else:
            # Weight factorization: kernel = g * v
            g = torch.exp(self.log_g)  # Ensure g > 0
            kernel = self.v * g.unsqueeze(0)  # Broadcasting: (in_features, 1) * (1, out_features)
            return F.linear(x, kernel.T, self.bias)

class FourierEmbedding(nn.Module):
    """
    Fourier feature embedding layer
    Converts JAX FourierEmbs to PyTorch equivalent
    """
    
    def __init__(self, input_dim: int, embed_dim: int, embed_scale: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed_scale = embed_scale
        
        # Random Fourier features: y = [cos(2π * scale * x @ W), sin(2π * scale * x @ W)]
        # W ~ Normal(0, scale^2)
        self.register_buffer('W', torch.randn(input_dim, embed_dim // 2) * embed_scale)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        # Output shape: (batch, embed_dim)
        # Ensure W is on the same device as x
        W = self.W.to(x.device)
        projected = torch.matmul(x, W)  # (batch, embed_dim // 2)
        fourier_features = torch.cat([
            torch.cos(projected),
            torch.sin(projected)
        ], dim=-1)
        return fourier_features

class PeriodEmbedding(nn.Module):
    """
    Periodic embedding layer (not used for cavity flow, but included for completeness)
    Converts JAX PeriodEmbs to PyTorch equivalent
    """
    
    def __init__(self, periods: Tuple[float, ...], axes: Tuple[int, ...], trainable: Tuple[bool, ...]):
        super().__init__()
        self.axes = axes
        
        # Initialize period parameters
        for i, (period, is_trainable) in enumerate(zip(periods, trainable)):
            if is_trainable:
                self.register_parameter(f'period_{i}', nn.Parameter(torch.tensor(period)))
            else:
                self.register_buffer(f'period_{i}', torch.tensor(period))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply periodic embeddings to specified axes
        y_components = []
        
        for i in range(x.shape[-1]):
            if i in self.axes:
                axis_idx = self.axes.index(i)
                period = getattr(self, f'period_{axis_idx}')
                xi = x[..., i]
                y_components.extend([torch.cos(period * xi), torch.sin(period * xi)])
            else:
                y_components.append(x[..., i])
        
        return torch.stack(y_components, dim=-1)

class ModifiedMlp(nn.Module):
    """
    Modified MLP (Pirate Network) architecture
    Converts JAX ModifiedMlp to PyTorch equivalent
    
    Key features:
    - Residual adaptive connections: x = x * u + (1-x) * v
    - Optional Fourier embeddings
    - Weight factorization for improved training
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        
        # Activation function
        self.activation = self._get_activation(config.activation)
        
        # Input processing
        input_dim = 2  # (x, y) coordinates
        
        # Optional Fourier embeddings
        if config.fourier_emb is not None:
            self.fourier_emb = FourierEmbedding(
                input_dim=input_dim,
                embed_dim=config.fourier_emb["embed_dim"],
                embed_scale=config.fourier_emb["embed_scale"]
            )
            current_dim = config.fourier_emb["embed_dim"]
        else:
            self.fourier_emb = None
            current_dim = input_dim
        
        # Optional periodic embeddings (not used for cavity flow)
        if config.periodicity is not None:
            self.period_emb = PeriodEmbedding(**config.periodicity)
        else:
            self.period_emb = None
        
        # First layer: create two parallel paths u and v
        self.u_layer = Dense(current_dim, self.hidden_dim, config.reparam)
        self.v_layer = Dense(current_dim, self.hidden_dim, config.reparam)
        
        # Hidden layers with residual adaptive connections
        # First layer: current_dim -> hidden_dim, rest: hidden_dim -> hidden_dim
        layer_dims = [current_dim] + [self.hidden_dim] * (self.num_layers - 1)
        self.hidden_layers = nn.ModuleList([
            Dense(layer_dims[i], self.hidden_dim, config.reparam) 
            for i in range(self.num_layers)
        ])
        
        # Output layer
        self.output_layer = Dense(self.hidden_dim, self.out_dim, config.reparam)
        
        current_dim = self.hidden_dim  # For hidden layers
    
    def _get_activation(self, activation_name: str):
        """Get activation function by name"""
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "gelu": F.gelu,
            "swish": F.silu,  # PyTorch calls swish 'silu'
            "sigmoid": torch.sigmoid,
            "sin": torch.sin,
        }
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")
        return activations[activation_name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Pirate network
        Based on JAX ModifiedMlp implementation
        
        Args:
            x: Input coordinates (batch, 2) for (x, y)
            
        Returns:
            Output (batch, 3) for (u, v, p) - velocity and pressure
        """
        # Input processing
        if self.period_emb is not None:
            x = self.period_emb(x)
        
        if self.fourier_emb is not None:
            x = self.fourier_emb(x)
        
        # Create parallel paths u and v from input (key innovation of Pirate networks)
        u = self.u_layer(x)
        v = self.v_layer(x)
        u = self.activation(u)
        v = self.activation(v)
        
        # Main path through hidden layers with adaptive residual connections
        # Start with processed input, but need to transform to hidden_dim first
        # The first transformation is done by a hidden layer
        current = x  # Processed input (current_dim)
        
        for layer in self.hidden_layers:
            # Apply layer to current state
            current = layer(current)
            current = self.activation(current)
            
            # Key Pirate network operation: adaptive residual connection  
            # This is applied to the layer output, creating the adaptive pathway
            current = current * u + (1 - current) * v
        
        # Output layer
        output = self.output_layer(current)
        return output

class StandardMlp(nn.Module):
    """
    Standard MLP for comparison
    Converts JAX Mlp to PyTorch equivalent
    """
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Input processing (same as ModifiedMlp)
        input_dim = 2
        if config.fourier_emb is not None:
            self.fourier_emb = FourierEmbedding(
                input_dim=input_dim,
                embed_dim=config.fourier_emb["embed_dim"],
                embed_scale=config.fourier_emb["embed_scale"]
            )
            current_dim = config.fourier_emb["embed_dim"]
        else:
            self.fourier_emb = None
            current_dim = input_dim
        
        # Build layers
        layers = []
        for i in range(config.num_layers):
            layers.append(Dense(current_dim, config.hidden_dim, config.reparam))
            layers.append(self._get_activation(config.activation))
            current_dim = config.hidden_dim
        
        layers.append(Dense(current_dim, config.out_dim, config.reparam))
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, activation_name: str):
        """Get activation function as nn.Module"""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "sigmoid": nn.Sigmoid(),
        }
        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")
        return activations[activation_name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier_emb is not None:
            x = self.fourier_emb(x)
        return self.network(x)

def create_network(config: NetworkConfig) -> nn.Module:
    """Factory function to create network based on config"""
    if config.arch_name == "ModifiedMlp":
        return ModifiedMlp(config)
    elif config.arch_name == "Mlp":
        return StandardMlp(config)
    else:
        raise ValueError(f"Unsupported architecture: {config.arch_name}")

if __name__ == "__main__":
    # Test network creation and forward pass
    from settings import Config
    
    config = Config()
    print("Testing Pirate Network...")
    print(f"Network config: {config.network.arch_name}")
    
    # Create network
    net = create_network(config.network)
    print(f"✓ Network created: {sum(p.numel() for p in net.parameters())} parameters")
    
    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 2)  # (x, y) coordinates
    
    with torch.no_grad():
        output = net(x)
    
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
    print(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✓ Pirate Network test completed!")