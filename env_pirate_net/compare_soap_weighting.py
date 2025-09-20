"""
Compare SOAP Optimizer with Different Weighting Schemes
Direct comparison: SOAP + Fixed vs SOAP + Gradient Norm Weighting
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from settings import Config
from training import PINNTrainer

def create_soap_config(weighting_scheme):
    """Create SOAP configuration with specified weighting scheme"""
    config = Config()

    # Common SOAP settings
    config.training.optimizer = "SOAP"
    config.training.learning_rate = 3e-3
    config.training.betas = (0.95, 0.95)
    config.training.weight_decay = 0.01
    config.training.precondition_frequency = 10

    # Training parameters (shorter for comparison)
    config.training.epochs = 1000
    config.training.log_every = 100
    config.training.n_collocation = 1500
    config.training.n_boundary = 300

    # Configure weighting scheme
    if weighting_scheme == "fixed":
        config.weighting.scheme = "fixed"
        config.physics.loss_weights = {
            "physics": 1.0,
            "boundary": 10.0,    # Manual tuning
            "continuity": 1.0,
            "momentum_x": 1.0,
            "momentum_y": 1.0,
        }
    elif weighting_scheme == "grad_norm":
        config.weighting.scheme = "grad_norm"
        config.weighting.grad_norm = {
            "alpha": 0.9,
            "update_every": 50,
            "eps": 1e-8
        }
        # Start with equal weights for fair comparison
        config.physics.loss_weights = {
            "physics": 1.0,
            "boundary": 1.0,
            "continuity": 1.0,
            "momentum_x": 1.0,
            "momentum_y": 1.0,
        }

    return config

def train_soap_scheme(scheme_name, seed=42):
    """Train SOAP with specified weighting scheme"""
    print(f"\nTraining SOAP + {scheme_name.upper()} Weighting...")
    print("-" * 50)

    # Set seed for reproducible comparison
    torch.manual_seed(seed)

    # Create configuration
    config = create_soap_config(scheme_name)

    # Create trainer
    trainer = PINNTrainer(config)

    # Track metrics
    history = {
        'epochs': [],
        'total_loss': [],
        'physics_loss': [],
        'boundary_loss': [],
        'momentum_x': [],
        'momentum_y': [],
        'continuity': [],
        'training_time': []
    }

    # Add weight tracking for gradient norm
    if scheme_name == "grad_norm":
        history.update({
            'weight_ru': [],
            'weight_rv': [],
            'weight_rc': [],
            'weight_u_bc': [],
            'weight_v_bc': []
        })

    start_time = time.time()

    # Training loop
    for epoch in range(config.training.epochs):
        epoch_start = time.time()
        loss_dict = trainer.train_step()
        epoch_time = time.time() - epoch_start

        # Store metrics
        history['epochs'].append(epoch)
        history['total_loss'].append(loss_dict['total'])
        history['physics_loss'].append(loss_dict.get('physics_total', 0))
        history['boundary_loss'].append(loss_dict.get('boundary_total', 0))
        history['momentum_x'].append(loss_dict['ru'])
        history['momentum_y'].append(loss_dict['rv'])
        history['continuity'].append(loss_dict['rc'])
        history['training_time'].append(epoch_time)

        # Store weights for gradient norm
        if scheme_name == "grad_norm":
            history['weight_ru'].append(loss_dict.get('weight_ru', 1.0))
            history['weight_rv'].append(loss_dict.get('weight_rv', 1.0))
            history['weight_rc'].append(loss_dict.get('weight_rc', 1.0))
            history['weight_u_bc'].append(loss_dict.get('weight_u_bc', 1.0))
            history['weight_v_bc'].append(loss_dict.get('weight_v_bc', 1.0))

        # Log progress
        if epoch % config.training.log_every == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss_dict['total']:.3e}, "
                  f"Time = {epoch_time:.3f}s")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")

    return trainer, history, total_time

def compare_soap_schemes():
    """Compare SOAP with different weighting schemes"""
    print("SOAP OPTIMIZER WEIGHTING SCHEME COMPARISON")
    print("=" * 60)
    print("Comparing:")
    print("1. SOAP + Fixed Weighting (Manual tuning)")
    print("2. SOAP + Gradient Norm Weighting (Automatic)")
    print()

    # Train both schemes
    results = {}

    # SOAP + Fixed
    trainer_fixed, history_fixed, time_fixed = train_soap_scheme("fixed", seed=42)
    results['fixed'] = {
        'trainer': trainer_fixed,
        'history': history_fixed,
        'time': time_fixed
    }

    # SOAP + Gradient Norm
    trainer_grad, history_grad, time_grad = train_soap_scheme("grad_norm", seed=42)
    results['grad_norm'] = {
        'trainer': trainer_grad,
        'history': history_grad,
        'time': time_grad
    }

    return results

def visualize_soap_comparison(results):
    """Create comprehensive comparison visualization"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 20))

    # Get data
    fixed_history = results['fixed']['history']
    grad_history = results['grad_norm']['history']

    epochs = fixed_history['epochs']

    # Plot 1: Total Loss Comparison
    ax1.semilogy(epochs, fixed_history['total_loss'], 'r-', linewidth=2,
                label='SOAP + Fixed Weighting')
    ax1.semilogy(epochs, grad_history['total_loss'], 'b-', linewidth=2,
                label='SOAP + Gradient Norm')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Total Loss Convergence Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Physics vs Boundary Loss Balance
    # Fixed weighting
    fixed_physics_ratio = np.array(fixed_history['physics_loss']) / np.array(fixed_history['total_loss'])
    fixed_boundary_ratio = np.array(fixed_history['boundary_loss']) / np.array(fixed_history['total_loss'])

    # Gradient norm weighting
    grad_physics_ratio = np.array(grad_history['physics_loss']) / np.array(grad_history['total_loss'])
    grad_boundary_ratio = np.array(grad_history['boundary_loss']) / np.array(grad_history['total_loss'])

    ax2.plot(epochs, fixed_physics_ratio, 'r-', alpha=0.7, label='Fixed: Physics/Total')
    ax2.plot(epochs, fixed_boundary_ratio, 'r--', alpha=0.7, label='Fixed: Boundary/Total')
    ax2.plot(epochs, grad_physics_ratio, 'b-', alpha=0.7, label='Grad Norm: Physics/Total')
    ax2.plot(epochs, grad_boundary_ratio, 'b--', alpha=0.7, label='Grad Norm: Boundary/Total')
    ax2.axhline(y=0.5, color='k', linestyle=':', alpha=0.5, label='Perfect Balance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Ratio')
    ax2.set_title('Loss Balance Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # Plot 3: Individual Loss Components - Fixed
    ax3.semilogy(epochs, fixed_history['momentum_x'], 'r-', label='X-Momentum', alpha=0.8)
    ax3.semilogy(epochs, fixed_history['momentum_y'], 'g-', label='Y-Momentum', alpha=0.8)
    ax3.semilogy(epochs, fixed_history['continuity'], 'b-', label='Continuity', alpha=0.8)
    ax3.semilogy(epochs, fixed_history['boundary_loss'], 'm-', label='Boundary', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Components (log scale)')
    ax3.set_title('SOAP + Fixed: Individual Components')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Individual Loss Components - Gradient Norm
    ax4.semilogy(epochs, grad_history['momentum_x'], 'r-', label='X-Momentum', alpha=0.8)
    ax4.semilogy(epochs, grad_history['momentum_y'], 'g-', label='Y-Momentum', alpha=0.8)
    ax4.semilogy(epochs, grad_history['continuity'], 'b-', label='Continuity', alpha=0.8)
    ax4.semilogy(epochs, grad_history['boundary_loss'], 'm-', label='Boundary', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Components (log scale)')
    ax4.set_title('SOAP + Gradient Norm: Individual Components')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Weight Evolution (Gradient Norm only)
    ax5.plot(epochs, grad_history['weight_ru'], 'r-', label='X-Momentum Weight', linewidth=2)
    ax5.plot(epochs, grad_history['weight_rv'], 'g-', label='Y-Momentum Weight', linewidth=2)
    ax5.plot(epochs, grad_history['weight_rc'], 'b-', label='Continuity Weight', linewidth=2)
    ax5.plot(epochs, grad_history['weight_u_bc'], 'm-', label='U-Boundary Weight', linewidth=2)
    ax5.plot(epochs, grad_history['weight_v_bc'], 'c-', label='V-Boundary Weight', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Adaptive Weight')
    ax5.set_title('Adaptive Weight Evolution (Gradient Norm)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Training Speed Comparison
    fixed_cumtime = np.cumsum(fixed_history['training_time'])
    grad_cumtime = np.cumsum(grad_history['training_time'])

    ax6.plot(epochs, fixed_cumtime, 'r-', linewidth=2, label='SOAP + Fixed')
    ax6.plot(epochs, grad_cumtime, 'b-', linewidth=2, label='SOAP + Gradient Norm')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Cumulative Time (seconds)')
    ax6.set_title('Training Speed Comparison')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    # Save comparison
    save_path = Path("results/soap_weighting_comparison.png")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {save_path}")

    return fig

def analyze_soap_comparison(results):
    """Analyze and compare SOAP performance with different weighting schemes"""
    print("\nSOAP WEIGHTING SCHEME COMPARISON ANALYSIS")
    print("=" * 60)

    fixed_history = results['fixed']['history']
    grad_history = results['grad_norm']['history']

    # Final loss comparison
    print("FINAL LOSS COMPARISON:")
    print("-" * 30)
    fixed_final = fixed_history['total_loss'][-1]
    grad_final = grad_history['total_loss'][-1]

    print(f"SOAP + Fixed:      {fixed_final:.4e}")
    print(f"SOAP + Grad Norm:  {grad_final:.4e}")

    if grad_final < fixed_final:
        improvement = ((fixed_final - grad_final) / fixed_final) * 100
        print(f"Gradient Norm is {improvement:.1f}% better")
    else:
        degradation = ((grad_final - fixed_final) / fixed_final) * 100
        print(f"Fixed is {degradation:.1f}% better")

    # Convergence rate comparison
    print(f"\nCONVERGENCE ANALYSIS:")
    print("-" * 25)

    # Calculate improvement over first 50% of training
    mid_point = len(fixed_history['total_loss']) // 2

    fixed_early_improvement = fixed_history['total_loss'][0] - fixed_history['total_loss'][mid_point]
    grad_early_improvement = grad_history['total_loss'][0] - grad_history['total_loss'][mid_point]

    print(f"Early convergence (first 50% epochs):")
    print(f"  Fixed:      {fixed_early_improvement:.3e}")
    print(f"  Grad Norm:  {grad_early_improvement:.3e}")

    # Training time comparison
    print(f"\nTRAINING EFFICIENCY:")
    print("-" * 25)
    fixed_time = results['fixed']['time']
    grad_time = results['grad_norm']['time']

    print(f"SOAP + Fixed time:     {fixed_time:.2f}s")
    print(f"SOAP + Grad Norm time: {grad_time:.2f}s")

    time_overhead = ((grad_time - fixed_time) / fixed_time) * 100
    print(f"Gradient norm overhead: {time_overhead:.1f}%")

    # Loss balance analysis
    print(f"\nLOSS BALANCE QUALITY:")
    print("-" * 25)

    # Final ratios
    fixed_physics_final = fixed_history['physics_loss'][-1] / fixed_history['total_loss'][-1]
    fixed_boundary_final = fixed_history['boundary_loss'][-1] / fixed_history['total_loss'][-1]

    grad_physics_final = grad_history['physics_loss'][-1] / grad_history['total_loss'][-1]
    grad_boundary_final = grad_history['boundary_loss'][-1] / grad_history['total_loss'][-1]

    print(f"Fixed - Physics/Boundary ratio: {fixed_physics_final:.3f}/{fixed_boundary_final:.3f}")
    print(f"Grad Norm - Physics/Boundary ratio: {grad_physics_final:.3f}/{grad_boundary_final:.3f}")

    fixed_balance = abs(fixed_physics_final - fixed_boundary_final)
    grad_balance = abs(grad_physics_final - grad_boundary_final)

    print(f"Balance quality (lower is better):")
    print(f"  Fixed balance diff:     {fixed_balance:.3f}")
    print(f"  Grad Norm balance diff: {grad_balance:.3f}")

    # Summary
    print(f"\nSUMMARY:")
    print("-" * 12)
    print("SOAP + Fixed Weighting:")
    print("  ✓ Predictable behavior")
    print("  ✓ Fast training (no overhead)")
    print("  ⚠ Requires manual weight tuning")
    print("  ⚠ May have suboptimal loss balance")

    print("\nSOAP + Gradient Norm Weighting:")
    print("  ✓ Automatic loss balancing")
    print("  ✓ No manual tuning required")
    print("  ✓ Better loss balance typically")
    print(f"  ⚠ Small computational overhead ({time_overhead:.1f}%)")

    return {
        'fixed_final_loss': fixed_final,
        'grad_final_loss': grad_final,
        'time_overhead_percent': time_overhead,
        'fixed_balance_quality': fixed_balance,
        'grad_balance_quality': grad_balance
    }

def main():
    """Main comparison function"""
    print("🔬 COMPREHENSIVE SOAP WEIGHTING COMPARISON")
    print("=" * 65)

    # Run comparison
    results = compare_soap_schemes()

    # Visualize results
    fig = visualize_soap_comparison(results)

    # Analyze results
    analysis = analyze_soap_comparison(results)

    # Show plots
    plt.show()

    print(f"\n🎯 RECOMMENDATION:")
    if analysis['grad_final_loss'] < analysis['fixed_final_loss']:
        print("Use SOAP + Gradient Norm Weighting for:")
        print("• Better convergence and loss balance")
        print("• Reduced need for manual tuning")
        print("• More robust training across problem types")
    else:
        print("Use SOAP + Fixed Weighting for:")
        print("• Maximum training speed")
        print("• When weights are well-known")

    return results, analysis

if __name__ == "__main__":
    results, analysis = main()