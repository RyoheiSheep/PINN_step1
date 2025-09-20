"""
SOAP Optimizer + Gradient Norm Weighting Implementation
Demonstrates advanced PINN training with SOAP optimizer and adaptive loss balancing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from settings import Config
from training import PINNTrainer

def create_soap_gradient_norm_config():
    """Create configuration for SOAP + Gradient Norm Weighting"""
    config = Config()

    # SOAP Optimizer Configuration
    config.training.optimizer = "SOAP"
    config.training.learning_rate = 3e-3
    config.training.betas = (0.95, 0.95)
    config.training.weight_decay = 0.01
    config.training.precondition_frequency = 10

    # Gradient Norm Weighting Configuration
    config.weighting.scheme = "grad_norm"
    config.weighting.grad_norm = {
        "alpha": 0.9,           # Exponential moving average momentum
        "update_every": 50,     # Update weights every 50 steps
        "eps": 1e-8            # Numerical stability
    }

    # Initial loss weights (will be adapted automatically)
    config.physics.loss_weights = {
        "physics": 1.0,
        "boundary": 1.0,     # Start equal, let gradient norm weighting adapt
        "continuity": 1.0,
        "momentum_x": 1.0,
        "momentum_y": 1.0,
    }

    # Training parameters
    config.training.epochs = 10
    config.training.log_every = 100
    config.training.n_collocation = 2000
    config.training.n_boundary = 400

    return config

def train_soap_gradient_norm():
    """Train PINN with SOAP optimizer and gradient norm weighting"""
    print("SOAP + Gradient Norm Weighting Training")
    print("=" * 50)

    # Create configuration
    config = create_soap_gradient_norm_config()

    print("Configuration:")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  SOAP Precondition Frequency: {config.training.precondition_frequency}")
    print(f"  Weighting Scheme: {config.weighting.scheme}")
    print(f"  Gradient Norm Update Frequency: {config.weighting.grad_norm['update_every']}")
    print(f"  Initial Weights: {config.physics.loss_weights}")
    print()

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create trainer
    trainer = PINNTrainer(config)

    # Track training metrics including adaptive weights
    training_history = {
        'epochs': [],
        'total_loss': [],
        'physics_loss': [],
        'boundary_loss': [],
        'momentum_x': [],
        'momentum_y': [],
        'continuity': [],
        # Adaptive weights
        'weight_ru': [],
        'weight_rv': [],
        'weight_rc': [],
        'weight_u_bc': [],
        'weight_v_bc': []
    }

    print("Training Progress:")
    print("Epoch | Total Loss | Physics  | Boundary | Weights (ru, rv, rc, u_bc, v_bc)")
    print("-" * 90)

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

        # Store adaptive weights
        training_history['weight_ru'].append(loss_dict.get('weight_ru', 1.0))
        training_history['weight_rv'].append(loss_dict.get('weight_rv', 1.0))
        training_history['weight_rc'].append(loss_dict.get('weight_rc', 1.0))
        training_history['weight_u_bc'].append(loss_dict.get('weight_u_bc', 1.0))
        training_history['weight_v_bc'].append(loss_dict.get('weight_v_bc', 1.0))

        # Log progress
        if epoch % config.training.log_every == 0:
            weights_str = f"({loss_dict.get('weight_ru', 1.0):.2f}, " \
                         f"{loss_dict.get('weight_rv', 1.0):.2f}, " \
                         f"{loss_dict.get('weight_rc', 1.0):.2f}, " \
                         f"{loss_dict.get('weight_u_bc', 1.0):.2f}, " \
                         f"{loss_dict.get('weight_v_bc', 1.0):.2f})"

            print(f"{epoch:5d} | {loss_dict['total']:8.2e} | "
                  f"{loss_dict['ru']:7.2e} | {loss_dict['u_bc']:7.2e} | {weights_str}")

    return trainer, training_history

def visualize_soap_gradient_norm_results(trainer, history):
    """Visualize training results for SOAP + Gradient Norm weighting"""

    # Create figure with more subplots to show weight evolution
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))

    epochs = history['epochs']

    # Plot 1: Total loss evolution
    ax1.semilogy(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('SOAP + Gradient Norm: Total Loss Evolution')
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

    # Plot 3: Adaptive weight evolution - Physics terms
    ax3.plot(epochs, history['weight_ru'], 'r-', label='X-Momentum Weight', linewidth=2)
    ax3.plot(epochs, history['weight_rv'], 'g-', label='Y-Momentum Weight', linewidth=2)
    ax3.plot(epochs, history['weight_rc'], 'b-', label='Continuity Weight', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Adaptive Weight')
    ax3.set_title('Physics Term Weight Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Adaptive weight evolution - Boundary terms
    ax4.plot(epochs, history['weight_u_bc'], 'r-', label='U Boundary Weight', linewidth=2)
    ax4.plot(epochs, history['weight_v_bc'], 'g-', label='V Boundary Weight', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Adaptive Weight')
    ax4.set_title('Boundary Term Weight Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Loss ratios showing automatic balancing
    boundary_ratio = np.array(history['boundary_loss']) / np.array(history['total_loss'])
    physics_ratio = np.array(history['physics_loss']) / np.array(history['total_loss'])

    ax5.plot(epochs, boundary_ratio, 'r-', label='Boundary/Total', linewidth=2)
    ax5.plot(epochs, physics_ratio, 'b-', label='Physics/Total', linewidth=2)
    ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Perfect Balance')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss Ratio')
    ax5.set_title('Automatic Loss Balance (Gradient Norm Effect)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim(0, 1)

    # Plot 6: Solution visualization
    solution = trainer.physics.evaluate_solution(trainer.network, grid_resolution=50)

    # Velocity magnitude
    speed = solution['speed']
    im = ax6.contourf(solution['x'], solution['y'], speed, levels=20, cmap='viridis')
    ax6.contour(solution['x'], solution['y'], speed, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('Final Solution: Velocity Magnitude')
    ax6.set_aspect('equal')
    plt.colorbar(im, ax=ax6, label='Speed')

    plt.tight_layout()

    # Save figure
    save_path = Path("results/soap_gradient_norm_results.png")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {save_path}")

    return fig

def analyze_soap_gradient_norm_performance(history):
    """Analyze SOAP + Gradient Norm weighting performance"""
    print("\nSOAP + Gradient Norm Weighting Performance Analysis:")
    print("=" * 55)

    final_losses = {
        'Total': history['total_loss'][-1],
        'Physics': history['physics_loss'][-1],
        'Boundary': history['boundary_loss'][-1],
        'X-Momentum': history['momentum_x'][-1],
        'Y-Momentum': history['momentum_y'][-1],
        'Continuity': history['continuity'][-1]
    }

    final_weights = {
        'X-Momentum': history['weight_ru'][-1],
        'Y-Momentum': history['weight_rv'][-1],
        'Continuity': history['weight_rc'][-1],
        'U-Boundary': history['weight_u_bc'][-1],
        'V-Boundary': history['weight_v_bc'][-1]
    }

    print("Final Loss Values:")
    for component, value in final_losses.items():
        print(f"  {component:12s}: {value:.4e}")

    print("\nFinal Adaptive Weights:")
    for component, weight in final_weights.items():
        print(f"  {component:12s}: {weight:.4f}")

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

    # Analyze weight adaptation
    weight_changes = {}
    for key in final_weights:
        weight_key = f"weight_{key.lower().replace('-', '_')}"
        if weight_key.replace('weight_', '') in ['ru', 'rv', 'rc', 'u_bc', 'v_bc']:
            initial_weight = history[f"weight_{key.lower().replace('-', '_').replace('boundary', 'bc')}"][0]
            final_weight = final_weights[key]
            change = final_weight - initial_weight
            weight_changes[key] = change

    print(f"\nWeight Adaptation Analysis:")
    for component, change in weight_changes.items():
        direction = "increased" if change > 0 else "decreased"
        print(f"  {component:12s}: {direction} by {abs(change):.3f}")

    # Check loss balance
    final_boundary_ratio = final_losses['Boundary'] / final_losses['Total']
    final_physics_ratio = final_losses['Physics'] / final_losses['Total']

    print(f"\nAutomatic Loss Balance Analysis:")
    print(f"  Boundary/Total:   {final_boundary_ratio:.3f}")
    print(f"  Physics/Total:    {final_physics_ratio:.3f}")
    print(f"  Balance Quality:  ", end="")

    balance_diff = abs(final_boundary_ratio - final_physics_ratio)
    if balance_diff < 0.2:
        print("Excellent (well balanced)")
    elif balance_diff < 0.4:
        print("Good (reasonably balanced)")
    else:
        print("Needs improvement")

    return final_losses, final_weights

def main():
    """Main function for SOAP + Gradient Norm Weighting demonstration"""
    print("SOAP OPTIMIZER + GRADIENT NORM WEIGHTING DEMONSTRATION")
    print("=" * 65)
    print("This demo shows advanced PINN training with:")
    print("• SOAP optimizer (2nd order adaptive method)")
    print("• Gradient norm weighting (automatic loss balancing)")
    print("• No manual tuning required!")
    print()

    # Train with SOAP + Gradient Norm weighting
    trainer, history = train_soap_gradient_norm()

    # Analyze results
    final_losses, final_weights = analyze_soap_gradient_norm_performance(history)

    # Visualize results
    fig = visualize_soap_gradient_norm_results(trainer, history)

    # Show the plot
    plt.show()

    print("\nSOAP + Gradient Norm Weighting Characteristics:")
    print("✓ Stable convergence with SOAP's adaptive preconditioning")
    print("✓ Automatic loss balancing via gradient norm weighting")
    print("✓ No manual weight tuning required")
    print("✓ Prevents gradient pathologies")
    print("✓ Robust across different problem scales")
    print("✓ Research-grade training with monitoring capabilities")

    return trainer, history, final_losses, final_weights

if __name__ == "__main__":
    # Run the demonstration
    trainer, history, losses, weights = main()