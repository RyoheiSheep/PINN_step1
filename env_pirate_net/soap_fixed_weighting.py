"""
SOAP Optimizer + Fixed Weighting Implementation
Demonstrates traditional PINN training with SOAP optimizer and manual loss weights
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from settings import Config
from training import PINNTrainer

def create_soap_fixed_config():
    """Create configuration for SOAP + Fixed Weighting"""
    config = Config()

    # SOAP Optimizer Configuration
    config.training.optimizer = "SOAP"
    config.training.learning_rate = 3e-3
    config.training.betas = (0.95, 0.95)
    config.training.weight_decay = 0.01
    config.training.precondition_frequency = 10

    # Fixed Weighting Configuration
    config.weighting.scheme = "fixed"

    # Manual loss weights (requires expert tuning)
    config.physics.loss_weights = {
        "physics": 1.0,      # Navier-Stokes residuals
        "boundary": 10.0,    # Boundary conditions (often need higher weight)
        "continuity": 1.0,   # Mass conservation
        "momentum_x": 1.0,   # X-momentum equation
        "momentum_y": 1.0,   # Y-momentum equation
    }

    # Training parameters
    config.training.epochs = 2000
    config.training.log_every = 100
    config.training.n_collocation = 2000
    config.training.n_boundary = 400

    return config

def train_soap_fixed():
    """Train PINN with SOAP optimizer and fixed weighting"""
    print("SOAP + Fixed Weighting Training")
    print("=" * 50)

    # Create configuration
    config = create_soap_fixed_config()

    print("Configuration:")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  SOAP Precondition Frequency: {config.training.precondition_frequency}")
    print(f"  Weighting Scheme: {config.weighting.scheme}")
    print(f"  Loss Weights: {config.physics.loss_weights}")
    print()

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create trainer
    trainer = PINNTrainer(config)

    # Track training metrics
    training_history = {
        'epochs': [],
        'total_loss': [],
        'physics_loss': [],
        'boundary_loss': [],
        'momentum_x': [],
        'momentum_y': [],
        'continuity': []
    }

    print("Training Progress:")
    print("Epoch | Total Loss | Physics  | Boundary | Mom-X    | Mom-Y    | Continuity")
    print("-" * 80)

    # Training loop
    for epoch in range(config.training.epochs):
        loss_dict = trainer.train_step()

        # Store metrics
        training_history['epochs'].append(epoch)
        training_history['total_loss'].append(loss_dict['total'])
        training_history['physics_loss'].append(loss_dict.get('physics_total', 0))
        training_history['boundary_loss'].append(loss_dict.get('boundary_total', 0))
        training_history['momentum_x'].append(loss_dict['ru'])
        training_history['momentum_y'].append(loss_dict['rv'])
        training_history['continuity'].append(loss_dict['rc'])

        # Log progress
        if epoch % config.training.log_every == 0:
            print(f"{epoch:5d} | {loss_dict['total']:8.2e} | "
                  f"{loss_dict['ru']:7.2e} | {loss_dict['u_bc']:7.2e} | "
                  f"{loss_dict['ru']:7.2e} | {loss_dict['rv']:7.2e} | "
                  f"{loss_dict['rc']:7.2e}")

    return trainer, training_history

def visualize_soap_fixed_results(trainer, history):
    """Visualize training results for SOAP + Fixed weighting"""

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = history['epochs']

    # Plot 1: Total loss evolution
    ax1.semilogy(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('SOAP + Fixed Weighting: Total Loss Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Individual loss components
    ax2.semilogy(epochs, history['momentum_x'], 'r-', label='X-Momentum', alpha=0.8)
    ax2.semilogy(epochs, history['momentum_y'], 'g-', label='Y-Momentum', alpha=0.8)
    ax2.semilogy(epochs, history['continuity'], 'b-', label='Continuity', alpha=0.8)
    ax2.semilogy(epochs, history['boundary_loss'], 'm-', label='Boundary', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Components (log scale)')
    ax2.set_title('Individual Loss Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Loss ratios (to check balance)
    boundary_ratio = np.array(history['boundary_loss']) / np.array(history['total_loss'])
    physics_ratio = np.array(history['physics_loss']) / np.array(history['total_loss'])

    ax3.plot(epochs, boundary_ratio, 'r-', label='Boundary/Total', linewidth=2)
    ax3.plot(epochs, physics_ratio, 'b-', label='Physics/Total', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Ratio')
    ax3.set_title('Loss Component Ratios (Balance Check)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)

    # Plot 4: Solution visualization
    solution = trainer.physics.evaluate_solution(trainer.network, grid_resolution=50)

    # Velocity magnitude
    speed = solution['speed']
    im = ax4.contourf(solution['x'], solution['y'], speed, levels=20, cmap='viridis')
    ax4.contour(solution['x'], solution['y'], speed, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Final Solution: Velocity Magnitude')
    ax4.set_aspect('equal')
    plt.colorbar(im, ax=ax4, label='Speed')

    plt.tight_layout()

    # Save figure
    save_path = Path("results/soap_fixed_weighting_results.png")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {save_path}")

    return fig

def analyze_soap_fixed_performance(history):
    """Analyze SOAP + Fixed weighting performance"""
    print("\nSOAP + Fixed Weighting Performance Analysis:")
    print("=" * 50)

    final_losses = {
        'Total': history['total_loss'][-1],
        'Physics': history['physics_loss'][-1],
        'Boundary': history['boundary_loss'][-1],
        'X-Momentum': history['momentum_x'][-1],
        'Y-Momentum': history['momentum_y'][-1],
        'Continuity': history['continuity'][-1]
    }

    print("Final Loss Values:")
    for component, value in final_losses.items():
        print(f"  {component:12s}: {value:.4e}")

    # Calculate convergence metrics
    initial_loss = history['total_loss'][0]
    final_loss = history['total_loss'][-1]
    improvement = initial_loss - final_loss
    improvement_ratio = improvement / initial_loss

    print(f"\nConvergence Metrics:")
    print(f"  Initial Loss:     {initial_loss:.4e}")
    print(f"  Final Loss:       {final_loss:.4e}")
    print(f"  Improvement:      {improvement:.4e}")
    print(f"  Improvement %:    {improvement_ratio*100:.2f}%")

    # Check loss balance
    final_boundary_ratio = final_losses['Boundary'] / final_losses['Total']
    final_physics_ratio = final_losses['Physics'] / final_losses['Total']

    print(f"\nLoss Balance Analysis:")
    print(f"  Boundary/Total:   {final_boundary_ratio:.3f}")
    print(f"  Physics/Total:    {final_physics_ratio:.3f}")

    if final_boundary_ratio < 0.1:
        print("  ⚠️  Boundary loss may be too small (< 10% of total)")
    elif final_boundary_ratio > 0.9:
        print("  ⚠️  Boundary loss may be too large (> 90% of total)")
    else:
        print("  ✓  Loss balance seems reasonable")

    return final_losses

def main():
    """Main function for SOAP + Fixed Weighting demonstration"""
    print("SOAP OPTIMIZER + FIXED WEIGHTING DEMONSTRATION")
    print("=" * 60)
    print("This demo shows traditional PINN training with:")
    print("• SOAP optimizer (2nd order adaptive method)")
    print("• Fixed loss weighting (manual tuning required)")
    print("• Manual balance between physics and boundary losses")
    print()

    # Train with SOAP + Fixed weighting
    trainer, history = train_soap_fixed()

    # Analyze results
    final_losses = analyze_soap_fixed_performance(history)

    # Visualize results
    fig = visualize_soap_fixed_results(trainer, history)

    # Show the plot
    plt.show()

    print("\nSOAP + Fixed Weighting Characteristics:")
    print("✓ Stable convergence with SOAP's adaptive preconditioning")
    print("✓ Predictable behavior with fixed weights")
    print("⚠️ Requires manual tuning of loss weights")
    print("⚠️ May not achieve optimal loss balance")
    print("⚠️ Sensitive to problem-specific weight choices")

    return trainer, history, final_losses

if __name__ == "__main__":
    # Run the demonstration
    trainer, history, results = main()