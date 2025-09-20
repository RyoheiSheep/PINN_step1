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
import datetime
import os
import uuid

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

def train_soap_fixed(runtime_id):
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

    # Training loop with best model tracking
    best_loss = float('inf')
    best_model_state = None

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

        # Track best model
        current_loss = loss_dict['total']
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = trainer.network.state_dict().copy()

        # Log progress
        if epoch % config.training.log_every == 0:
            print(f"{epoch:5d} | {loss_dict['total']:8.2e} | "
                  f"{loss_dict['ru']:7.2e} | {loss_dict['u_bc']:7.2e} | "
                  f"{loss_dict['ru']:7.2e} | {loss_dict['rv']:7.2e} | "
                  f"{loss_dict['rc']:7.2e}")

    # Save best model
    save_best_model(trainer, best_model_state, best_loss, runtime_id)

    return trainer, training_history

def save_best_model(trainer, best_model_state, best_loss, runtime_id):
    """Save the best model during training"""
    results_dir = Path(f"results/run_{runtime_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "best_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'best_loss': best_loss,
        'runtime_id': runtime_id,
        'model_config': {
            'hidden_dim': trainer.network.hidden_dim,
            'output_dim': trainer.network.out_dim,
            'num_layers': trainer.network.num_layers,
            'config': trainer.network.config.__dict__
        }
    }, model_path)

    print(f"\nBest model saved to: {model_path}")
    print(f"Best loss: {best_loss:.6e}")

def create_comprehensive_visualizations(trainer, runtime_id):
    """Create and save comprehensive visualizations including streamlines, pressure, and velocity fields"""
    results_dir = Path(f"results/run_{runtime_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate solution on high-resolution grid
    solution = trainer.physics.evaluate_solution(trainer.network, grid_resolution=100)

    # Create comprehensive visualization figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Velocity magnitude with streamlines
    ax = axes[0, 0]
    speed = solution['speed']
    im1 = ax.contourf(solution['x'], solution['y'], speed, levels=20, cmap='viridis')

    # Add streamlines
    u = solution['u']
    v = solution['v']
    ax.streamplot(solution['x'], solution['y'], u, v, color='white', linewidth=0.8, density=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Velocity Magnitude with Streamlines')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax, label='Speed')

    # 2. Pressure field
    ax = axes[0, 1]
    pressure = solution['p']
    im2 = ax.contourf(solution['x'], solution['y'], pressure, levels=20, cmap='RdBu_r')
    ax.contour(solution['x'], solution['y'], pressure, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pressure Field')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, label='Pressure')

    # 3. U velocity component
    ax = axes[0, 2]
    im3 = ax.contourf(solution['x'], solution['y'], u, levels=20, cmap='RdBu_r')
    ax.contour(solution['x'], solution['y'], u, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('U Velocity Component')
    ax.set_aspect('equal')
    plt.colorbar(im3, ax=ax, label='U Velocity')

    # 4. V velocity component
    ax = axes[1, 0]
    im4 = ax.contourf(solution['x'], solution['y'], v, levels=20, cmap='RdBu_r')
    ax.contour(solution['x'], solution['y'], v, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('V Velocity Component')
    ax.set_aspect('equal')
    plt.colorbar(im4, ax=ax, label='V Velocity')

    # 5. Vorticity
    ax = axes[1, 1]
    # Calculate vorticity (curl of velocity)
    dx = solution['x'][0, 1] - solution['x'][0, 0]
    dy = solution['y'][1, 0] - solution['y'][0, 0]

    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    vorticity = dv_dx - du_dy

    im5 = ax.contourf(solution['x'], solution['y'], vorticity, levels=20, cmap='RdBu_r')
    ax.contour(solution['x'], solution['y'], vorticity, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vorticity Field')
    ax.set_aspect('equal')
    plt.colorbar(im5, ax=ax, label='Vorticity')

    # 6. Streamlines only (for clarity)
    ax = axes[1, 2]
    strm = ax.streamplot(solution['x'], solution['y'], u, v, color=speed, linewidth=1.5, cmap='viridis', density=2)
    plt.colorbar(strm.lines, ax=ax, label='Speed')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Streamlines (colored by speed)')
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save comprehensive visualization
    comprehensive_path = results_dir / "comprehensive_solution.png"
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive solution saved to: {comprehensive_path}")

    return fig, solution

def visualize_soap_fixed_results(trainer, history, runtime_id):
    """Visualize training results for SOAP + Fixed weighting"""
    results_dir = Path(f"results/run_{runtime_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # Save training history figure
    save_path = results_dir / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history saved to: {save_path}")

    # Create and save comprehensive visualizations
    comp_fig, solution = create_comprehensive_visualizations(trainer, runtime_id)

    return fig, comp_fig, solution

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
        print("  !  Boundary loss may be too small (< 10% of total)")
    elif final_boundary_ratio > 0.9:
        print("  !  Boundary loss may be too large (> 90% of total)")
    else:
        print("  +  Loss balance seems reasonable")

    return final_losses

def main():
    """Main function for SOAP + Fixed Weighting demonstration"""
    # Generate unique runtime ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    runtime_id = f"{timestamp}_{unique_id}"

    print("SOAP OPTIMIZER + FIXED WEIGHTING DEMONSTRATION")
    print("=" * 60)
    print("This demo shows traditional PINN training with:")
    print("- SOAP optimizer (2nd order adaptive method)")
    print("- Fixed loss weighting (manual tuning required)")
    print("- Manual balance between physics and boundary losses")
    print(f"- Runtime ID: {runtime_id}")
    print()

    # Train with SOAP + Fixed weighting
    trainer, history = train_soap_fixed(runtime_id)

    # Analyze results
    final_losses = analyze_soap_fixed_performance(history)

    # Visualize results
    training_fig, comprehensive_fig, solution = visualize_soap_fixed_results(trainer, history, runtime_id)

    # Save runtime metadata
    results_dir = Path(f"results/run_{runtime_id}")
    metadata_path = results_dir / "run_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Runtime ID: {runtime_id}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Method: SOAP + Fixed Weighting\n")
        f.write(f"Final Loss: {history['total_loss'][-1]:.6e}\n")
        f.write(f"Training Epochs: {len(history['epochs'])}\n")
        f.write(f"Best Loss: {min(history['total_loss']):.6e}\n")

    print(f"\nAll results saved in: {results_dir}")
    print(f"Files created:")
    print(f"  - best_model.pt")
    print(f"  - training_history.png")
    print(f"  - comprehensive_solution.png")
    print(f"  - run_metadata.txt")

    # Show the plots
    plt.show()

    print("\nSOAP + Fixed Weighting Characteristics:")
    print("+ Stable convergence with SOAP's adaptive preconditioning")
    print("+ Predictable behavior with fixed weights")
    print("! Requires manual tuning of loss weights")
    print("! May not achieve optimal loss balance")
    print("! Sensitive to problem-specific weight choices")

    return trainer, history, final_losses, runtime_id

if __name__ == "__main__":
    # Run the demonstration
    trainer, history, results, runtime_id = main()