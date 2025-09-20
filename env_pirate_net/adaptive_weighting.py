"""
Adaptive Loss Weighting for Physics-Informed Neural Networks
Implements gradient norm weighting and Neural Tangent Kernel (NTK) weighting
Based on JAX-PI implementation
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import defaultdict

class GradientNormWeighting:
    """
    Automatic loss balancing using gradient norm weighting

    Based on the principle that loss terms should have similar gradient magnitudes
    to prevent one term from dominating the optimization.

    Reference: "Understanding and Mitigating Gradient Flow Pathologies in PINNs"
    """

    def __init__(self,
                 alpha: float = 0.9,
                 update_every: int = 100,
                 eps: float = 1e-8):
        """
        Args:
            alpha: Momentum factor for exponential moving average
            update_every: Update frequency for gradient norm computation
            eps: Small constant for numerical stability
        """
        self.alpha = alpha
        self.update_every = update_every
        self.eps = eps
        self.step_count = 0

        # Storage for gradient norms and weights
        self.grad_norms_ema = {}
        self.weights = {}
        self.initialized = False

    def compute_gradient_norms(self,
                              network: nn.Module,
                              loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute gradient norms for each loss component

        Args:
            network: Neural network
            loss_dict: Dictionary of individual loss components

        Returns:
            Dictionary of gradient norms
        """
        grad_norms = {}

        for loss_name, loss_value in loss_dict.items():
            if loss_name.endswith('_total'):
                continue  # Skip total losses

            # Compute gradients w.r.t. this loss component
            gradients = torch.autograd.grad(
                outputs=loss_value,
                inputs=network.parameters(),
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )

            # Compute gradient norm
            grad_norm = 0.0
            for grad in gradients:
                if grad is not None:
                    grad_norm += torch.sum(grad ** 2).item()

            grad_norms[loss_name] = np.sqrt(grad_norm)

        return grad_norms

    def update_weights(self,
                      network: nn.Module,
                      loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update loss weights based on gradient norms

        Args:
            network: Neural network
            loss_dict: Dictionary of individual loss components

        Returns:
            Updated weights dictionary
        """
        self.step_count += 1

        # Compute gradient norms
        grad_norms = self.compute_gradient_norms(network, loss_dict)

        # Initialize or update exponential moving averages
        if not self.initialized:
            self.grad_norms_ema = grad_norms.copy()
            # Initialize weights to 1.0
            self.weights = {k: 1.0 for k in grad_norms.keys()}
            self.initialized = True
        else:
            # Update EMAs
            for key in grad_norms:
                if key in self.grad_norms_ema:
                    self.grad_norms_ema[key] = (
                        self.alpha * self.grad_norms_ema[key] +
                        (1 - self.alpha) * grad_norms[key]
                    )
                else:
                    self.grad_norms_ema[key] = grad_norms[key]

        # Update weights every N steps
        if self.step_count % self.update_every == 0:
            # Target: balance gradient norms
            # Use the average gradient norm as target
            avg_grad_norm = np.mean(list(self.grad_norms_ema.values()))

            for key in self.grad_norms_ema:
                # Weight inversely proportional to gradient norm
                current_norm = self.grad_norms_ema[key]
                if current_norm > self.eps:
                    self.weights[key] = avg_grad_norm / (current_norm + self.eps)
                else:
                    self.weights[key] = 1.0

        return self.weights.copy()


class NTKWeighting:
    """
    Neural Tangent Kernel based loss weighting

    Computes NTK values for different loss components and uses them
    to balance the optimization dynamics.

    Reference: "When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective"
    """

    def __init__(self,
                 update_every: int = 1000,
                 alpha: float = 0.9,
                 eps: float = 1e-8):
        """
        Args:
            update_every: Update frequency for NTK computation (expensive)
            alpha: Momentum factor for exponential moving average
            eps: Small constant for numerical stability
        """
        self.update_every = update_every
        self.alpha = alpha
        self.eps = eps
        self.step_count = 0

        # Storage for NTK values and weights
        self.ntk_values_ema = {}
        self.weights = {}
        self.initialized = False

    def compute_ntk_diagonal(self,
                           network: nn.Module,
                           points: torch.Tensor,
                           output_idx: int = 0) -> torch.Tensor:
        """
        Compute diagonal NTK values for given points

        Args:
            network: Neural network
            points: Input points
            output_idx: Which output dimension to compute NTK for

        Returns:
            Diagonal NTK values
        """
        # Enable gradient computation
        points.requires_grad_(True)

        # Forward pass
        outputs = network(points)

        if outputs.dim() > 1:
            outputs = outputs[:, output_idx]

        # Compute Jacobian w.r.t. network parameters
        jacobians = []
        for i in range(outputs.shape[0]):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[i] = 1.0

            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=network.parameters(),
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )

            # Flatten and concatenate gradients
            flat_grads = []
            for grad in grads:
                if grad is not None:
                    flat_grads.append(grad.view(-1))

            if flat_grads:
                jacobian = torch.cat(flat_grads)
                jacobians.append(jacobian)

        if not jacobians:
            return torch.zeros(outputs.shape[0], device=outputs.device)

        # Stack jacobians: (n_points, n_params)
        jacobian_matrix = torch.stack(jacobians)

        # Compute diagonal NTK: diag(J @ J^T)
        ntk_diagonal = torch.sum(jacobian_matrix ** 2, dim=1)

        return ntk_diagonal

    def compute_ntk_values(self,
                          network: nn.Module,
                          collocation_points: torch.Tensor,
                          boundary_points: torch.Tensor) -> Dict[str, float]:
        """
        Compute mean NTK values for different domains

        Args:
            network: Neural network
            collocation_points: Interior domain points
            boundary_points: Boundary points

        Returns:
            Dictionary of mean NTK values
        """
        ntk_values = {}

        # NTK for physics terms (interior domain)
        with torch.enable_grad():
            # For velocity components
            ntk_u_physics = self.compute_ntk_diagonal(network, collocation_points, output_idx=0)
            ntk_v_physics = self.compute_ntk_diagonal(network, collocation_points, output_idx=1)

            ntk_values['ru'] = torch.mean(ntk_u_physics).item()
            ntk_values['rv'] = torch.mean(ntk_v_physics).item()
            ntk_values['rc'] = (ntk_values['ru'] + ntk_values['rv']) / 2  # Average for continuity

            # NTK for boundary conditions
            ntk_u_boundary = self.compute_ntk_diagonal(network, boundary_points, output_idx=0)
            ntk_v_boundary = self.compute_ntk_diagonal(network, boundary_points, output_idx=1)

            ntk_values['u_bc'] = torch.mean(ntk_u_boundary).item()
            ntk_values['v_bc'] = torch.mean(ntk_v_boundary).item()

        return ntk_values

    def update_weights(self,
                      network: nn.Module,
                      collocation_points: torch.Tensor,
                      boundary_points: torch.Tensor) -> Dict[str, float]:
        """
        Update loss weights based on NTK values

        Args:
            network: Neural network
            collocation_points: Interior domain points
            boundary_points: Boundary points

        Returns:
            Updated weights dictionary
        """
        self.step_count += 1

        # Only compute NTK every N steps (expensive operation)
        if self.step_count % self.update_every == 0:
            # Compute NTK values
            ntk_values = self.compute_ntk_values(network, collocation_points, boundary_points)

            # Initialize or update exponential moving averages
            if not self.initialized:
                self.ntk_values_ema = ntk_values.copy()
                # Initialize weights to 1.0
                self.weights = {k: 1.0 for k in ntk_values.keys()}
                self.initialized = True
            else:
                # Update EMAs
                for key in ntk_values:
                    if key in self.ntk_values_ema:
                        self.ntk_values_ema[key] = (
                            self.alpha * self.ntk_values_ema[key] +
                            (1 - self.alpha) * ntk_values[key]
                        )
                    else:
                        self.ntk_values_ema[key] = ntk_values[key]

            # Update weights: inverse of NTK values (higher NTK = lower weight)
            max_ntk = max(self.ntk_values_ema.values())
            for key in self.ntk_values_ema:
                ntk_value = self.ntk_values_ema[key]
                if ntk_value > self.eps:
                    self.weights[key] = max_ntk / (ntk_value + self.eps)
                else:
                    self.weights[key] = 1.0

        return self.weights.copy()


class AdaptiveWeightingManager:
    """
    Manager class for different adaptive weighting schemes
    """

    def __init__(self,
                 scheme: str = "fixed",
                 grad_norm_config: Optional[Dict] = None,
                 ntk_config: Optional[Dict] = None):
        """
        Args:
            scheme: Weighting scheme ("fixed", "grad_norm", "ntk")
            grad_norm_config: Configuration for gradient norm weighting
            ntk_config: Configuration for NTK weighting
        """
        self.scheme = scheme

        if scheme == "grad_norm":
            config = grad_norm_config or {}
            self.weighter = GradientNormWeighting(**config)
        elif scheme == "ntk":
            config = ntk_config or {}
            self.weighter = NTKWeighting(**config)
        elif scheme == "fixed":
            self.weighter = None
        else:
            raise ValueError(f"Unknown weighting scheme: {scheme}")

    def get_weights(self,
                   network: nn.Module,
                   loss_dict: Dict[str, torch.Tensor],
                   collocation_points: Optional[torch.Tensor] = None,
                   boundary_points: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Get current loss weights

        Args:
            network: Neural network
            loss_dict: Dictionary of individual loss components
            collocation_points: Interior domain points (for NTK)
            boundary_points: Boundary points (for NTK)

        Returns:
            Dictionary of loss weights
        """
        if self.scheme == "fixed":
            # Return unit weights for all loss components
            return {k: 1.0 for k in loss_dict.keys() if not k.endswith('_total')}

        elif self.scheme == "grad_norm":
            return self.weighter.update_weights(network, loss_dict)

        elif self.scheme == "ntk":
            if collocation_points is None or boundary_points is None:
                raise ValueError("NTK weighting requires collocation_points and boundary_points")
            return self.weighter.update_weights(network, collocation_points, boundary_points)

        else:
            raise ValueError(f"Unknown weighting scheme: {self.scheme}")