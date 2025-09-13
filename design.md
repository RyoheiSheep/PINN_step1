# Technical Design Document

## Overview

This document outlines the technical design for implementing a Physics-Informed Neural Network (PINN) system using Pirate network architecture and SOAP optimizer for 2D incompressible cavity flow.

## System Architecture

### High-Level Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   settings.py   │───▶│  PirateNetwork   │───▶│  CavityFlow     │
│   (Config)      │    │  (PyTorch)       │    │  (Physics)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Visualizer    │◀───│  TrainingLoop    │◀───│  SOAPOptimizer  │
│   (Plots)       │    │  (Main)          │    │  (Optimizer)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Module Design

#### 1. Configuration Module (`settings.py`)
```python
class NetworkConfig:
    # Network architecture
    hidden_layers: int = 4
    hidden_dim: int = 128
    activation: str = "tanh"
    
class TrainingConfig:
    # Training parameters
    epochs: int = 10000
    learning_rate: float = 1e-3
    n_collocation: int = 2000
    n_boundary: int = 400
    
class PhysicsConfig:
    # Problem-specific parameters
    reynolds_number: float = 100.0
    lid_velocity: float = 1.0
    domain_size: tuple = (1.0, 1.0)
```

#### 2. Network Architecture (`pirate_network.py`)
```python
class ResidualBlock(nn.Module):
    """Residual adaptive block from Pirate network"""
    
class PirateNetwork(nn.Module):
    """Main Pirate network architecture in PyTorch"""
    def __init__(self, config: NetworkConfig):
        # Convert from JAX implementation
    
    def forward(self, x):
        # Input: (batch, 2) -> (x, y) coordinates
        # Output: (batch, 3) -> (u, v, p) velocity and pressure
```

#### 3. Physics Module (`cavity_flow.py`)
```python
class CavityFlowProblem:
    """2D incompressible cavity flow physics"""
    
    def generate_collocation_points(self, n_points):
        """Generate interior domain points"""
        
    def generate_boundary_points(self, n_points):
        """Generate boundary points for all walls"""
        
    def physics_loss(self, coords, network_output):
        """Compute Navier-Stokes residuals"""
        # Continuity: ∇·u = 0
        # Momentum: u·∇u = -∇p + (1/Re)∇²u
        
    def boundary_loss(self, coords, network_output):
        """Compute boundary condition violations"""
        # No-slip: u=v=0 on walls
        # Lid velocity: u=1, v=0 on top wall
```

#### 4. Training Module (`training.py`)
```python
class PINNTrainer:
    """Main training orchestrator"""
    
    def __init__(self, network, optimizer, physics_problem):
        self.network = network
        self.optimizer = optimizer  # SOAP
        self.physics = physics_problem
        
    def train_step(self):
        """Single training iteration"""
        
    def train(self, epochs):
        """Full training loop with logging"""
```

#### 5. Visualization Module (`visualizer.py`)
```python
class FlowVisualizer:
    """Flow field visualization"""
    
    def plot_velocity_field(self, network):
        """Vector plot of velocity field"""
        
    def plot_streamlines(self, network):
        """Streamline visualization"""
        
    def plot_pressure_contours(self, network):
        """Pressure contour plot"""
        
    def plot_loss_history(self, losses):
        """Training loss curves"""
```

## Data Flow Design

### Training Data Flow
```
1. settings.py → Configuration loading
2. CavityFlowProblem → Generate training points
3. PirateNetwork → Forward pass on points
4. CavityFlowProblem → Compute physics + boundary loss
5. SOAPOptimizer → Backpropagation and parameter update
6. TrainingLoop → Loss logging and visualization
```

### Inference Data Flow
```
1. Trained network → Load checkpoint
2. Grid generation → Create evaluation mesh
3. Network inference → Predict (u, v, p) on grid
4. FlowVisualizer → Generate plots and analysis
```

## Implementation Details

### Network Architecture Conversion
- **Source**: JAX-based Pirate network in `pirate_net/` directory
- **Target**: PyTorch implementation with equivalent functionality
- **Key Components**:
  - Residual adaptive blocks
  - Weight initialization schemes
  - Activation functions and normalization

### SOAP Optimizer Integration
- **Source**: PyTorch SOAP implementation in `SOAP/soap.py`
- **Integration**: Direct usage with PyTorch autograd
- **Configuration**:
  ```python
  optimizer = SOAP(
      network.parameters(),
      lr=3e-3,
      betas=(0.95, 0.95),
      weight_decay=0.01,
      precondition_frequency=10
  )
  ```

### Physics Loss Implementation
- **Automatic Differentiation**: Use `torch.autograd.grad()` for computing derivatives
- **Navier-Stokes Equations**:
  ```
  Continuity: ∂u/∂x + ∂v/∂y = 0
  X-momentum: u∂u/∂x + v∂u/∂y = -∂p/∂x + (1/Re)(∂²u/∂x² + ∂²u/∂y²)
  Y-momentum: u∂v/∂x + v∂v/∂y = -∂p/∂y + (1/Re)(∂²v/∂x² + ∂²v/∂y²)
  ```

### Boundary Conditions
- **Bottom, Left, Right walls**: u = v = 0 (no-slip)
- **Top wall (lid)**: u = 1, v = 0 (moving lid)
- **Implementation**: Separate loss terms for each boundary segment

## File Structure

```
PINN_step1/
├── settings.py              # Configuration
├── pirate_network.py        # Network architecture
├── cavity_flow.py           # Physics problem
├── training.py              # Training loop
├── visualizer.py            # Visualization
├── main.py                  # Entry point
├── utils.py                 # Helper functions
└── checkpoints/             # Model saves
```

## Performance Considerations

### Memory Optimization
- Batch processing of collocation points
- Gradient checkpointing for large networks
- Efficient tensor operations

### Numerical Stability
- Proper weight initialization
- Gradient clipping if needed
- Loss scaling for mixed precision

### Monitoring
- Loss component tracking (physics vs boundary)
- Gradient norm monitoring
- Training progress visualization

## Testing Strategy

### Unit Tests
- Network forward pass correctness
- Physics loss computation accuracy
- Boundary condition enforcement

### Integration Tests
- End-to-end training pipeline
- Configuration loading and validation
- Visualization output quality

### Validation Tests
- Known analytical solutions (if available)
- Physics conservation properties
- Convergence behavior analysis