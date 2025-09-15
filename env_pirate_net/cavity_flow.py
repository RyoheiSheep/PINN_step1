"""
Cavity Flow Physics Implementation
Based on official JAX LDC example: pirate_net/examples/ldc/

Implements 2D incompressible Navier-Stokes equations:
- Continuity: ∇·u = 0  
- X-momentum: u∂u/∂x + v∂u/∂y = -∂p/∂x + ν∇²u
- Y-momentum: u∂v/∂x + v∂v/∂y = -∂p/∂y + ν∇²v

Boundary conditions for lid-driven cavity:
- Top wall (lid): u = U_lid, v = 0
- Other walls: u = 0, v = 0 (no-slip)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

from settings import PhysicsConfig

class CavityFlowProblem:
    """
    2D lid-driven cavity flow problem implementation
    Based on JAX reference: pirate_net/examples/ldc/models.py and utils.py
    """

    def __init__(self, config: PhysicsConfig, training_config=None):
        self.config = config
        self.training_config = training_config  # Need this for n_boundary
        self.re = config.reynolds_number
        self.nu = config.viscosity  # Computed in config post_init
        self.lid_velocity = config.lid_velocity

        # Domain bounds
        self.x_bounds, self.y_bounds = config.domain_bounds
        self.x_min, self.x_max = self.x_bounds
        self.y_min, self.y_max = self.y_bounds

        # Pre-generate boundary points (following JAX reference)
        # Use n_boundary from training config if available
        if training_config:
            num_pts_per_side = training_config.n_boundary // 4
        else:
            num_pts_per_side = 64  # fallback default
        self.boundary_points = self.generate_boundary_points(num_pts_per_side)

        # Boundary condition values (following JAX reference setup)
        self._setup_boundary_conditions()
    
    def generate_collocation_points(self, n_points: int, random: bool = True) -> torch.Tensor:
        """
        Generate collocation points in the interior domain
        
        Args:
            n_points: Number of points to generate
            random: If True, random sampling; if False, regular grid
            
        Returns:
            Points tensor of shape (n_points, 2) for (x, y) coordinates
        """
        if random:
            # Random sampling in interior (avoiding boundaries)
            x = torch.rand(n_points) * (self.x_max - self.x_min) + self.x_min
            y = torch.rand(n_points) * (self.y_max - self.y_min) + self.y_min
        else:
            # Regular grid
            n_side = int(np.sqrt(n_points))
            x = torch.linspace(self.x_min, self.x_max, n_side)
            y = torch.linspace(self.y_min, self.y_max, n_side)
            x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
            x = x_grid.flatten()[:n_points]
            y = y_grid.flatten()[:n_points]
        
        points = torch.stack([x, y], dim=1)
        return points
    
    def generate_boundary_points(self, num_pts_per_side: int = 64, eps: float = 0.01) -> torch.Tensor:
        """
        Generate boundary points on cavity walls
        Based on JAX reference: sample_points_on_square_boundary()
        
        Args:
            num_pts_per_side: Points per boundary side
            eps: Small offset to avoid corner singularities
            
        Returns:
            Boundary points tensor of shape (4*num_pts_per_side, 2)
        """
        # Top wall (y=1, x: 0→1) - moving lid
        top_x = torch.linspace(self.x_min, self.x_max, num_pts_per_side)
        top_y = torch.ones_like(top_x) * self.y_max
        top = torch.stack([top_x, top_y], dim=1)
        
        # Bottom wall (y=0, x: 0→1) - no-slip
        bottom_x = torch.linspace(self.x_min, self.x_max, num_pts_per_side)
        bottom_y = torch.ones_like(bottom_x) * self.y_min
        bottom = torch.stack([bottom_x, bottom_y], dim=1)
        
        # Left wall (x=0, y: 0→1-eps) - no-slip, avoid corner
        left_y = torch.linspace(self.y_min, self.y_max - eps, num_pts_per_side)
        left_x = torch.ones_like(left_y) * self.x_min
        left = torch.stack([left_x, left_y], dim=1)
        
        # Right wall (x=1, y: 0→1-eps) - no-slip, avoid corner
        right_y = torch.linspace(self.y_min, self.y_max - eps, num_pts_per_side)
        right_x = torch.ones_like(right_y) * self.x_max
        right = torch.stack([right_x, right_y], dim=1)
        
        # Combine all boundary points
        boundary_points = torch.cat([top, bottom, left, right], dim=0)
        return boundary_points
    
    def _setup_boundary_conditions(self):
        """
        Setup boundary condition values following JAX reference
        """
        num_pts = len(self.boundary_points) // 4
        total_pts = len(self.boundary_points)
        
        # Initialize all boundary velocities to zero
        self.u_bc = torch.zeros(total_pts)
        self.v_bc = torch.zeros(total_pts)
        
        # Set top wall (lid) velocity: u = lid_velocity, v = 0
        # First num_pts points are top wall
        self.u_bc[:num_pts] = self.lid_velocity
        # v_bc already zero for all points
    
    def compute_physics_residuals(self, network: nn.Module, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Navier-Stokes residuals at given coordinates
        Based on JAX reference: r_net() function
        
        Args:
            network: Physics-informed neural network
            coords: Coordinate tensor (batch, 2) for (x, y)
            
        Returns:
            Tuple of (ru, rv, rc) residuals for momentum and continuity
        """
        coords.requires_grad_(True)
        x, y = coords[:, 0], coords[:, 1]
        
        # Forward pass: get (u, v, p)
        output = network(coords)
        u, v, p = output[:, 0], output[:, 1], output[:, 2]
        
        # Compute first derivatives using autograd
        # Following JAX: (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(neural_net, argnums=(1, 2))
        u_grad = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        v_grad = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
        p_grad = torch.autograd.grad(p.sum(), coords, create_graph=True)[0]
        
        u_x, u_y = u_grad[:, 0], u_grad[:, 1]
        v_x, v_y = v_grad[:, 0], v_grad[:, 1]
        p_x, p_y = p_grad[:, 0], p_grad[:, 1]
        
        # Compute second derivatives for Laplacian
        # Following JAX: u_hessian = hessian(u_net, argnums=(1, 2))
        u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
        
        v_xx = torch.autograd.grad(v_x.sum(), coords, create_graph=True)[0][:, 0]
        v_yy = torch.autograd.grad(v_y.sum(), coords, create_graph=True)[0][:, 1]
        
        # Navier-Stokes residuals (following JAX reference exactly)
        # X-momentum: ru = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        ru = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        
        # Y-momentum: rv = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)  
        rv = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        
        # Continuity: rc = u_x + v_y (incompressibility)
        rc = u_x + v_y
        
        return ru, rv, rc
    
    def compute_boundary_loss(self, network: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute boundary condition losses
        Following JAX reference: losses() function boundary part
        
        Args:
            network: Physics-informed neural network
            
        Returns:
            Dictionary with boundary loss components
        """
        # FIXED: Removed torch.no_grad() to allow proper gradient flow for boundary conditions
        boundary_coords = self.boundary_points
        
        # Forward pass on boundary points
        output = network(boundary_coords)
        u_pred = output[:, 0]
        v_pred = output[:, 1]
        
        # Compute boundary losses (following JAX reference)
        u_bc_loss = torch.mean((u_pred - self.u_bc) ** 2)
        v_bc_loss = torch.mean((v_pred - self.v_bc) ** 2)
        
        return {
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "boundary_total": u_bc_loss + v_bc_loss
        }
    
    def compute_physics_loss(self, network: nn.Module, collocation_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based losses from PDE residuals
        Following JAX reference: losses() function physics part
        
        Args:
            network: Physics-informed neural network  
            collocation_points: Interior domain points
            
        Returns:
            Dictionary with physics loss components
        """
        # Compute PDE residuals
        ru, rv, rc = self.compute_physics_residuals(network, collocation_points)
        
        # Mean squared residuals (following JAX reference)
        ru_loss = torch.mean(ru ** 2)
        rv_loss = torch.mean(rv ** 2) 
        rc_loss = torch.mean(rc ** 2)
        
        return {
            "ru": ru_loss,          # X-momentum residual
            "rv": rv_loss,          # Y-momentum residual  
            "rc": rc_loss,          # Continuity residual
            "physics_total": ru_loss + rv_loss + rc_loss
        }
    
    def compute_total_loss(self, network: nn.Module, collocation_points: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total weighted loss combining physics and boundary terms
        
        Args:
            network: Physics-informed neural network
            collocation_points: Interior domain points for physics
            
        Returns:
            Tuple of (total_loss, loss_dict) 
        """
        # Compute individual loss components
        boundary_losses = self.compute_boundary_loss(network)
        physics_losses = self.compute_physics_loss(network, collocation_points)
        
        # Apply loss weighting from config
        weights = self.config.loss_weights
        
        weighted_losses = {
            "u_bc": weights["boundary"] * boundary_losses["u_bc"],
            "v_bc": weights["boundary"] * boundary_losses["v_bc"],
            "ru": weights["momentum_x"] * physics_losses["ru"],
            "rv": weights["momentum_y"] * physics_losses["rv"],
            "rc": weights["continuity"] * physics_losses["rc"],
        }
        
        # Total loss
        total_loss = sum(weighted_losses.values())
        
        # Complete loss dictionary for monitoring
        loss_dict = {
            **weighted_losses,
            "boundary_total": boundary_losses["boundary_total"],
            "physics_total": physics_losses["physics_total"],
            "total": total_loss
        }
        
        return total_loss, loss_dict
    
    def evaluate_solution(self, network: nn.Module, grid_resolution: int = 100) -> Dict[str, np.ndarray]:
        """
        Evaluate trained network on regular grid for visualization
        
        Args:
            network: Trained physics-informed neural network
            grid_resolution: Grid resolution for evaluation
            
        Returns:
            Dictionary with solution fields on grid
        """
        # Create evaluation grid
        x = torch.linspace(self.x_min, self.x_max, grid_resolution)
        y = torch.linspace(self.y_min, self.y_max, grid_resolution)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Evaluate network
        with torch.no_grad():
            output = network(grid_points)
            u = output[:, 0].reshape(grid_resolution, grid_resolution)
            v = output[:, 1].reshape(grid_resolution, grid_resolution)
            p = output[:, 2].reshape(grid_resolution, grid_resolution)
        
        return {
            "x": X.numpy(),
            "y": Y.numpy(), 
            "u": u.numpy(),
            "v": v.numpy(),
            "p": p.numpy(),
            "speed": np.sqrt(u.numpy()**2 + v.numpy()**2)
        }
    
    def plot_solution(self, solution: Dict[str, np.ndarray], save_path: Optional[str] = None):
        """
        Plot cavity flow solution
        
        Args:
            solution: Solution dictionary from evaluate_solution()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Cavity Flow Solution (Re={self.re})', fontsize=16)
        
        x, y = solution['x'], solution['y']
        
        # U velocity
        im1 = axes[0,0].contourf(x, y, solution['u'], levels=20, cmap='RdBu_r')
        axes[0,0].set_title('U Velocity')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0,0])
        
        # V velocity  
        im2 = axes[0,1].contourf(x, y, solution['v'], levels=20, cmap='RdBu_r')
        axes[0,1].set_title('V Velocity')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Pressure
        im3 = axes[1,0].contourf(x, y, solution['p'], levels=20, cmap='viridis')
        axes[1,0].set_title('Pressure')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1,0])
        
        # Speed with streamlines
        speed = solution['speed']
        axes[1,1].contourf(x, y, speed, levels=20, cmap='plasma')
        # streamplot needs 1D arrays for x and y coordinates
        x_1d = x[0, :]  # First row of X (all x values)
        y_1d = y[:, 0]  # First column of Y (all y values)
        axes[1,1].streamplot(x_1d, y_1d, solution['u'], solution['v'], color='white', linewidth=0.8)
        axes[1,1].set_title('Speed + Streamlines')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig

if __name__ == "__main__":
    # Test cavity flow problem setup
    from settings import Config
    
    print("Testing Cavity Flow Problem...")
    
    config = Config()
    physics = CavityFlowProblem(config.physics, config.training)
    
    print(f"✓ Physics problem created")
    print(f"  Reynolds number: {physics.re}")
    print(f"  Viscosity: {physics.nu:.6f}")
    print(f"  Domain: {physics.x_bounds} × {physics.y_bounds}")
    
    # Test point generation
    collocation_pts = physics.generate_collocation_points(1000)
    boundary_pts = physics.boundary_points
    
    print(f"✓ Point generation:")
    print(f"  Collocation points: {collocation_pts.shape}")
    print(f"  Boundary points: {boundary_pts.shape}")
    print(f"  Boundary conditions: u_bc={physics.u_bc[:4]}, v_bc={physics.v_bc[:4]}")
    
    # Test with dummy network
    from pirate_network import create_network
    
    network = create_network(config.network)
    print(f"✓ Created test network")
    
    # Test loss computation
    total_loss, loss_dict = physics.compute_total_loss(network, collocation_pts)
    print(f"✓ Loss computation:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("✓ Cavity Flow Problem test completed!")