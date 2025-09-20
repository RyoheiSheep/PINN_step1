"""
Configuration system for PINN project with Pirate networks and SOAP optimizer.
Based on JAX reference implementation with PyTorch adaptations.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import torch

@dataclass
class NetworkConfig:
    """Network architecture configuration"""
    # Core architecture
    arch_name: str = "ModifiedMlp"  # Use ModifiedMlp (Pirate network)
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 3  # (u, v, p) for Navier-Stokes
    activation: str = "tanh"
    
    # Feature embeddings
    fourier_emb: Optional[Dict] = field(default_factory=lambda: {
        "embed_scale": 10.0,
        "embed_dim": 256
    })
    periodicity: Optional[Dict] = None  # Not used for cavity flow
    
    # Weight parameterization (from JAX reference)
    reparam: Optional[Dict] = field(default_factory=lambda: {
        "type": "weight_fact", 
        "mean": 1.0, 
        "stddev": 0.1
    })
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

@dataclass 
class TrainingConfig:
    """Training parameters configuration"""
    # Training loop
    epochs: int = 10000
    batch_size: int = 1024
    
    # Data sampling
    n_collocation: int = 2000  # Interior domain points
    n_boundary: int = 400      # Boundary points (100 per wall)
    
    # SOAP optimizer parameters (from reference)
    optimizer: str = "SOAP"
    learning_rate: float = 3e-3
    betas: Tuple[float, float] = (0.95, 0.95)
    weight_decay: float = 0.01
    precondition_frequency: int = 10
    
    # Learning rate scheduling
    lr_decay: bool = True
    decay_rate: float = 0.9
    decay_steps: int = 2000
    
    # Monitoring
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500
    
    # Early stopping
    patience: int = 1000
    min_delta: float = 1e-6

@dataclass
class PhysicsConfig:
    """Physics problem configuration for 2D cavity flow"""
    # Problem parameters
    reynolds_number: float = 100.0  # Re = 1/nu
    lid_velocity: float = 1.0       # Moving lid velocity
    
    # Domain geometry 
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))  # ((x_min, x_max), (y_min, y_max))
    
    # Loss weighting
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "physics": 1.0,     # Navier-Stokes residuals
        "boundary": 1.0,    # Boundary conditions  
        "continuity": 1.0,  # Mass conservation
        "momentum_x": 1.0,  # X-momentum
        "momentum_y": 1.0,  # Y-momentum
    })
    
    # Boundary condition types
    boundary_types: Dict[str, str] = field(default_factory=lambda: {
        "top": "moving_lid",    # u=lid_velocity, v=0
        "bottom": "no_slip",    # u=0, v=0
        "left": "no_slip",      # u=0, v=0  
        "right": "no_slip",     # u=0, v=0
    })

@dataclass
class AdaptiveWeightingConfig:
    """Configuration for adaptive loss weighting schemes"""
    # Weighting scheme: "fixed", "grad_norm", "ntk"
    scheme: str = "fixed"

    # Gradient norm weighting configuration
    grad_norm: Dict = field(default_factory=lambda: {
        "alpha": 0.9,           # EMA momentum factor
        "update_every": 100,    # Update frequency
        "eps": 1e-8,           # Numerical stability
    })

    # Neural Tangent Kernel weighting configuration
    ntk: Dict = field(default_factory=lambda: {
        "alpha": 0.9,           # EMA momentum factor
        "update_every": 1000,   # Update frequency (expensive)
        "eps": 1e-8,           # Numerical stability
    })

@dataclass
class Config:
    """Main configuration class combining all settings"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    weighting: AdaptiveWeightingConfig = field(default_factory=AdaptiveWeightingConfig)

    # Global settings
    seed: int = 42
    project_name: str = "PINN-CavityFlow-PirateNet"
    experiment_name: str = "default"
    save_dir: str = "./checkpoints"
    
    def __post_init__(self):
        """Validation and post-processing"""
        self._validate_config()
        self._setup_derived_params()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Network validation
        assert self.network.num_layers > 0, "Number of layers must be positive"
        assert self.network.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.network.out_dim == 3, "Output dimension must be 3 for (u,v,p)"
        assert self.network.activation in ["tanh", "relu", "gelu", "swish", "sigmoid"], f"Unsupported activation: {self.network.activation}"
        
        # Training validation  
        assert self.training.epochs > 0, "Epochs must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.training.betas[0] < 1 and 0 < self.training.betas[1] < 1, "Betas must be in (0,1)"
        assert self.training.n_collocation > 0, "Number of collocation points must be positive"
        assert self.training.n_boundary > 0, "Number of boundary points must be positive"
        
        # Physics validation
        assert self.physics.reynolds_number > 0, "Reynolds number must be positive"
        assert self.physics.lid_velocity != 0, "Lid velocity cannot be zero"
        
        # Domain bounds validation
        x_bounds, y_bounds = self.physics.domain_bounds
        assert x_bounds[1] > x_bounds[0], "Invalid x domain bounds"
        assert y_bounds[1] > y_bounds[0], "Invalid y domain bounds"
    
    def _setup_derived_params(self):
        """Set up derived parameters"""
        # Compute kinematic viscosity from Reynolds number
        self.physics.viscosity = self.physics.lid_velocity * (
            self.physics.domain_bounds[0][1] - self.physics.domain_bounds[0][0]
        ) / self.physics.reynolds_number
        
        # Set random seeds
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
    
    def summary(self) -> str:
        """Return a summary string of the configuration"""
        return f"""
PINN Configuration Summary:
==========================
Network: {self.network.arch_name} ({self.network.num_layers} layers, {self.network.hidden_dim} hidden)
Training: {self.training.epochs} epochs, lr={self.training.learning_rate}, batch={self.training.batch_size}
Physics: Re={self.physics.reynolds_number}, lid_vel={self.physics.lid_velocity}
Domain: {self.physics.domain_bounds}
Device: {self.network.device}
Optimizer: {self.training.optimizer}
        """.strip()

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default"""
    if config_path is None:
        return Config()
    
    # TODO: Implement config file loading (JSON/YAML)
    # For now, return default config
    return Config()

def create_reynolds_configs(re_values: List[float]) -> List[Config]:
    """Create multiple configurations for different Reynolds numbers"""
    configs = []
    for re in re_values:
        config = Config()
        config.physics.reynolds_number = re
        config.experiment_name = f"Re_{int(re)}"
        # Adjust training steps based on Reynolds number (higher Re needs more training)
        if re >= 1000:
            config.training.epochs = 20000
        elif re >= 400:
            config.training.epochs = 15000
        else:
            config.training.epochs = 10000
        configs.append(config)
    return configs

# Default configuration instance
default_config = Config()

if __name__ == "__main__":
    # Test configuration system
    config = Config()
    print(config.summary())
    print(f"\nValidation passed: ✓")
    print(f"Viscosity: {config.physics.viscosity:.6f}")
    print(f"Device: {config.network.device}")