"""
Training System for PINN with Pirate Networks and SOAP Optimizer
Integrates network, physics, and optimization components
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Import SOAP optimizer
import sys
sys.path.append('../SOAP')
from soap import SOAP

from settings import Config
from pirate_network import create_network
from cavity_flow import CavityFlowProblem
from adaptive_weighting import AdaptiveWeightingManager

class PINNTrainer:
    """
    Main training orchestrator for Physics-Informed Neural Networks
    Integrates Pirate network, SOAP optimizer, and cavity flow physics
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Setup device
        self.device = torch.device(config.network.device)
        print(f"Using device: {self.device}")
        
        # Create network
        self.network = create_network(config.network).to(self.device)
        print(f"Network created: {sum(p.numel() for p in self.network.parameters()):,} parameters")
        
        # Create physics problem
        self.physics = CavityFlowProblem(config.physics, config.training)
        print(f"Physics problem: Re={self.physics.re}, ν={self.physics.nu:.6f}")
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        print(f"Optimizer: {type(self.optimizer).__name__}")
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = defaultdict(list)
        self.start_time = None
        
        # Setup directories
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Generate training points
        self._generate_training_data()

        # Setup adaptive weighting
        self.weighting_manager = AdaptiveWeightingManager(
            scheme=config.weighting.scheme,
            grad_norm_config=config.weighting.grad_norm,
            ntk_config=config.weighting.ntk
        )
        print(f"Weighting scheme: {config.weighting.scheme}")
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create SOAP optimizer with configured parameters"""
        training_config = self.config.training
        
        if training_config.optimizer == "SOAP":
            return SOAP(
                self.network.parameters(),
                lr=training_config.learning_rate,
                betas=training_config.betas,
                weight_decay=training_config.weight_decay,
                precondition_frequency=training_config.precondition_frequency
            )
        elif training_config.optimizer == "Adam":
            return torch.optim.Adam(
                self.network.parameters(),
                lr=training_config.learning_rate,
                betas=training_config.betas,
                weight_decay=training_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config.optimizer}")
    
    def _generate_training_data(self):
        """Generate collocation and boundary points for training"""
        training_config = self.config.training
        
        # Generate collocation points (interior domain)
        self.collocation_points = self.physics.generate_collocation_points(
            training_config.n_collocation, random=True
        ).to(self.device)
        
        # Boundary points are pre-generated in physics problem
        self.boundary_points = self.physics.boundary_points.to(self.device)
        
        print(f"Training data: {len(self.collocation_points)} collocation + {len(self.boundary_points)} boundary points")
    
    def train_step(self) -> Dict[str, float]:
        """
        Single training step with adaptive loss weighting
        Returns loss dictionary for monitoring
        """
        self.network.train()
        self.optimizer.zero_grad()

        # First, compute individual loss components without weighting
        boundary_losses = self.physics.compute_boundary_loss(self.network)
        physics_losses = self.physics.compute_physics_loss(self.network, self.collocation_points)

        # Combine for adaptive weighting computation
        individual_losses = {
            "u_bc": boundary_losses["u_bc"],
            "v_bc": boundary_losses["v_bc"],
            "ru": physics_losses["ru"],
            "rv": physics_losses["rv"],
            "rc": physics_losses["rc"],
        }

        # Get adaptive weights
        adaptive_weights = self.weighting_manager.get_weights(
            network=self.network,
            loss_dict=individual_losses,
            collocation_points=self.collocation_points,
            boundary_points=self.boundary_points
        )

        # Compute total loss with adaptive weights
        total_loss, loss_dict = self.physics.compute_total_loss(
            self.network, self.collocation_points, adaptive_weights
        )

        # Backpropagation
        total_loss.backward()

        # Optimizer step
        self.optimizer.step()

        # Add weighting information to loss dict for logging
        weight_dict = {f"weight_{k}": v for k, v in adaptive_weights.items()}
        loss_dict.update(weight_dict)

        # Convert tensors to floats for logging
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

        return loss_dict
    
    def validate(self) -> Dict[str, float]:
        """
        Validation step (same as train for PINNs, but without parameter updates)
        Note: We still need gradients for physics residuals, so no torch.no_grad()
        """
        self.network.eval()
        
        # Generate fresh validation points (need gradients for physics)
        val_collocation = self.physics.generate_collocation_points(
            self.config.training.n_collocation // 2, random=True
        ).to(self.device)
        
        total_loss, loss_dict = self.physics.compute_total_loss(
            self.network, val_collocation
        )
        
        # Convert to floats
        loss_dict = {f"val_{k}": v.item() if torch.is_tensor(v) else v 
                    for k, v in loss_dict.items()}
        
        return loss_dict
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.training.epochs
        
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 60)
        
        self.start_time = time.time()
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training step
            train_losses = self.train_step()
            
            # Record training losses
            for key, value in train_losses.items():
                self.training_history[f"train_{key}"].append(value)
            
            # Validation step (periodic)
            if epoch % self.config.training.eval_every == 0:
                val_losses = self.validate()
                for key, value in val_losses.items():
                    self.training_history[key].append(value)
            
            # Logging
            if epoch % self.config.training.log_every == 0:
                self._log_progress(epoch, epochs, train_losses, val_losses if epoch % self.config.training.eval_every == 0 else None)
            
            # Model saving
            if epoch % self.config.training.save_every == 0 and epoch > 0:
                self.save_checkpoint(epoch)
            
            # Early stopping check
            current_loss = train_losses['total']
            if current_loss < self.best_loss - self.config.training.min_delta:
                self.best_loss = current_loss
                patience_counter = 0
                self.save_best_model()
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.training.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.config.training.patience})")
                break
        
        total_time = time.time() - self.start_time
        print(f"\nTraining completed in {total_time:.1f}s ({total_time/epochs:.3f}s/epoch)")
        print(f"Best loss: {self.best_loss:.6e}")
        
        return dict(self.training_history)
    
    def _log_progress(self, epoch: int, total_epochs: int, train_losses: Dict[str, float], val_losses: Optional[Dict[str, float]] = None):
        """Log training progress"""
        elapsed = time.time() - self.start_time
        eta = elapsed / (epoch + 1) * (total_epochs - epoch - 1)
        
        print(f"Epoch {epoch:5d}/{total_epochs} | "
              f"Time: {elapsed:6.1f}s | ETA: {eta:6.1f}s")
        
        # Training losses
        total_loss = train_losses['total']
        physics_loss = train_losses.get('physics_total', 0)
        boundary_loss = train_losses.get('boundary_total', 0)
        
        print(f"  Train Loss: {total_loss:.6e} "
              f"(Physics: {physics_loss:.6e}, Boundary: {boundary_loss:.6e})")
        
        # Detailed component losses
        components = ['ru', 'rv', 'rc', 'u_bc', 'v_bc']
        comp_str = " | ".join([f"{comp}: {train_losses.get(f'train_{comp}', train_losses.get(comp, 0)):.3e}" 
                              for comp in components])
        print(f"  Components: {comp_str}")
        
        # Validation losses
        if val_losses:
            val_total = val_losses['val_total']
            print(f"  Val Loss: {val_total:.6e}")
        
        print()
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:06d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self):
        """Save the best model"""
        model_path = self.save_dir / "best_model.pt"
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'epoch': self.current_epoch
        }, model_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.config.experiment_name}', fontsize=16)
        
        # Total loss
        train_total = self.training_history.get('train_total', [])
        val_total = self.training_history.get('val_total', [])
        
        axes[0,0].semilogy(train_total, label='Train', alpha=0.8)
        if val_total:
            val_epochs = np.linspace(0, len(train_total)-1, len(val_total))
            axes[0,0].semilogy(val_epochs, val_total, label='Val', alpha=0.8)
        axes[0,0].set_title('Total Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Physics vs Boundary loss
        physics_loss = self.training_history.get('train_physics_total', [])
        boundary_loss = self.training_history.get('train_boundary_total', [])
        
        if physics_loss and boundary_loss:
            axes[0,1].semilogy(physics_loss, label='Physics', alpha=0.8)
            axes[0,1].semilogy(boundary_loss, label='Boundary', alpha=0.8)
            axes[0,1].set_title('Physics vs Boundary Loss')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Component losses
        components = ['train_ru', 'train_rv', 'train_rc']
        for i, comp in enumerate(components):
            if comp in self.training_history:
                axes[1,0].semilogy(self.training_history[comp], 
                                  label=comp.replace('train_', '').upper(), alpha=0.8)
        axes[1,0].set_title('Physics Components')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Boundary components
        bc_components = ['train_u_bc', 'train_v_bc']
        for comp in bc_components:
            if comp in self.training_history:
                axes[1,1].semilogy(self.training_history[comp], 
                                  label=comp.replace('train_', '').upper(), alpha=0.8)
        axes[1,1].set_title('Boundary Components')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def evaluate_solution(self, grid_resolution: int = 50):
        """Evaluate trained network and create visualizations"""
        self.network.eval()
        
        with torch.no_grad():
            solution = self.physics.evaluate_solution(self.network, grid_resolution)
        
        # Create solution plots
        fig = self.physics.plot_solution(solution)
        
        return solution, fig

def train_cavity_flow(config: Config, resume_from: Optional[str] = None) -> PINNTrainer:
    """
    Main training function for cavity flow PINN
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained PINNTrainer instance
    """
    # Create trainer
    trainer = PINNTrainer(config)
    
    # Resume from checkpoint if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Run training
    history = trainer.train()
    
    # Plot results
    trainer.plot_training_history(
        save_path=trainer.save_dir / "training_history.png"
    )
    
    # Evaluate final solution
    solution, fig = trainer.evaluate_solution()
    fig.savefig(trainer.save_dir / "final_solution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Results saved to: {trainer.save_dir}")
    
    return trainer

if __name__ == "__main__":
    # Test training system
    print("Testing PINN Training System...")
    
    # Create test configuration
    from settings import Config
    
    config = Config()
    config.training.epochs = 100  # Short test
    config.training.log_every = 20
    config.experiment_name = "test_run"
    
    print("Configuration:")
    print(config.summary())
    
    # Run training test
    trainer = train_cavity_flow(config)
    
    print("✓ Training system test completed!")