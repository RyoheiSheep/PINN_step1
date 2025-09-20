"""
Comprehensive SOAP + Weighting Demonstration
Interactive demo showing both SOAP + Fixed and SOAP + Gradient Norm approaches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from settings import Config
from training import PINNTrainer

def create_interactive_demo():
    """Create interactive demo for SOAP weighting schemes"""
    print("🚀 INTERACTIVE SOAP WEIGHTING DEMONSTRATION")
    print("=" * 60)
    print("This demo will show you:")
    print("1. SOAP + Fixed Weighting (Traditional approach)")
    print("2. SOAP + Gradient Norm Weighting (Advanced approach)")
    print("3. Side-by-side comparison")
    print()

    # Get user choice
    while True:
        print("Choose demonstration:")
        print("1. SOAP + Fixed Weighting")
        print("2. SOAP + Gradient Norm Weighting")
        print("3. Compare both methods")
        print("4. Quick comparison (fast)")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice in ['1', '2', '3', '4']:
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

    return int(choice)

def run_soap_fixed_demo():
    """Run SOAP + Fixed weighting demo"""
    print("\n🔧 SOAP + FIXED WEIGHTING DEMO")
    print("=" * 40)
    print("Characteristics:")
    print("• Manual loss weight tuning required")
    print("• Predictable, stable behavior")
    print("• Fast training (no overhead)")
    print()

    # Configuration
    config = Config()
    config.training.optimizer = "SOAP"
    config.weighting.scheme = "fixed"
    config.training.epochs = 500
    config.training.log_every = 50

    # Manual weight tuning (this is the challenge!)
    config.physics.loss_weights = {
        "physics": 1.0,
        "boundary": 10.0,    # Often needs to be higher
        "continuity": 1.0,
        "momentum_x": 1.0,
        "momentum_y": 1.0,
    }

    print(f"Manual loss weights: {config.physics.loss_weights}")
    print("Starting training...")

    # Train
    torch.manual_seed(42)
    trainer = PINNTrainer(config)

    losses = []
    for epoch in range(config.training.epochs):
        loss_dict = trainer.train_step()
        losses.append(loss_dict['total'])

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss_dict['total']:.3e}")

    # Results
    print(f"\nFinal loss: {losses[-1]:.3e}")
    print("✓ Training completed with fixed weights")
    print("⚠️ Note: Weight tuning was done manually")

    return trainer, losses

def run_soap_gradient_norm_demo():
    """Run SOAP + Gradient Norm weighting demo"""
    print("\n🤖 SOAP + GRADIENT NORM WEIGHTING DEMO")
    print("=" * 45)
    print("Characteristics:")
    print("• Automatic loss balancing")
    print("• No manual tuning required")
    print("• Adaptive weight evolution")
    print()

    # Configuration
    config = Config()
    config.training.optimizer = "SOAP"
    config.weighting.scheme = "grad_norm"
    config.weighting.grad_norm = {
        "update_every": 25,  # Update frequently for demo
        "alpha": 0.9
    }
    config.training.epochs = 500
    config.training.log_every = 50

    # Start with equal weights - let algorithm adapt
    config.physics.loss_weights = {
        "physics": 1.0,
        "boundary": 1.0,     # Equal start
        "continuity": 1.0,
        "momentum_x": 1.0,
        "momentum_y": 1.0,
    }

    print(f"Initial weights: {config.physics.loss_weights}")
    print("Gradient norm weighting will adapt these automatically...")
    print("Starting training...")

    # Train
    torch.manual_seed(42)
    trainer = PINNTrainer(config)

    losses = []
    weights_history = []

    for epoch in range(config.training.epochs):
        loss_dict = trainer.train_step()
        losses.append(loss_dict['total'])

        # Track weight evolution
        weights = {
            'ru': loss_dict.get('weight_ru', 1.0),
            'u_bc': loss_dict.get('weight_u_bc', 1.0),
            'v_bc': loss_dict.get('weight_v_bc', 1.0)
        }
        weights_history.append(weights)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss_dict['total']:.3e}, "
                  f"Weights = ru:{weights['ru']:.2f}, u_bc:{weights['u_bc']:.2f}")

    # Results
    final_weights = weights_history[-1]
    print(f"\nFinal loss: {losses[-1]:.3e}")
    print(f"Final adaptive weights: {final_weights}")
    print("✓ Training completed with automatic balancing")
    print("🎯 Weights adapted automatically!")

    return trainer, losses, weights_history

def compare_soap_methods_quick():
    """Quick comparison of both methods"""
    print("\n⚡ QUICK SOAP COMPARISON")
    print("=" * 30)

    # Short configurations for speed
    epochs = 200
    torch.manual_seed(42)

    results = {}

    # Test Fixed
    print("Testing SOAP + Fixed...")
    config_fixed = Config()
    config_fixed.training.optimizer = "SOAP"
    config_fixed.weighting.scheme = "fixed"
    config_fixed.training.epochs = epochs
    config_fixed.training.log_every = 1000  # Quiet
    config_fixed.physics.loss_weights["boundary"] = 10.0

    trainer_fixed = PINNTrainer(config_fixed)
    losses_fixed = []

    for epoch in range(epochs):
        loss_dict = trainer_fixed.train_step()
        losses_fixed.append(loss_dict['total'])

    results['fixed'] = losses_fixed

    # Test Gradient Norm
    print("Testing SOAP + Gradient Norm...")
    torch.manual_seed(42)  # Same seed for fair comparison

    config_grad = Config()
    config_grad.training.optimizer = "SOAP"
    config_grad.weighting.scheme = "grad_norm"
    config_grad.weighting.grad_norm["update_every"] = 20
    config_grad.training.epochs = epochs
    config_grad.training.log_every = 1000  # Quiet

    trainer_grad = PINNTrainer(config_grad)
    losses_grad = []

    for epoch in range(epochs):
        loss_dict = trainer_grad.train_step()
        losses_grad.append(loss_dict['total'])

    results['grad_norm'] = losses_grad

    # Quick plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(epochs), losses_fixed, 'r-', label='SOAP + Fixed', linewidth=2)
    plt.semilogy(range(epochs), losses_grad, 'b-', label='SOAP + Gradient Norm', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (log scale)')
    plt.title('Quick SOAP Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    save_path = Path("results/soap_quick_comparison.png")
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Results
    print(f"\nQuick Comparison Results:")
    print(f"SOAP + Fixed final loss:      {losses_fixed[-1]:.3e}")
    print(f"SOAP + Gradient Norm final:   {losses_grad[-1]:.3e}")

    if losses_grad[-1] < losses_fixed[-1]:
        improvement = ((losses_fixed[-1] - losses_grad[-1]) / losses_fixed[-1]) * 100
        print(f"Gradient Norm is {improvement:.1f}% better")
    else:
        degradation = ((losses_grad[-1] - losses_fixed[-1]) / losses_fixed[-1]) * 100
        print(f"Fixed is {degradation:.1f}% better")

    return results

def run_full_comparison():
    """Run comprehensive comparison"""
    print("\n🔬 COMPREHENSIVE SOAP COMPARISON")
    print("=" * 40)
    print("Running full comparison - this will take longer...")

    # Import and run the comparison
    from compare_soap_weighting import compare_soap_schemes, analyze_soap_comparison

    results = compare_soap_schemes()
    analysis = analyze_soap_comparison(results)

    return results, analysis

def main():
    """Main demo function"""
    choice = create_interactive_demo()

    if choice == 1:
        # SOAP + Fixed demo
        trainer, losses = run_soap_fixed_demo()

        print(f"\n📊 SOAP + Fixed Summary:")
        print(f"• Final loss: {losses[-1]:.3e}")
        print(f"• Training approach: Manual weight tuning")
        print(f"• Best for: Known problems with established weights")

    elif choice == 2:
        # SOAP + Gradient Norm demo
        trainer, losses, weights = run_soap_gradient_norm_demo()

        print(f"\n📊 SOAP + Gradient Norm Summary:")
        print(f"• Final loss: {losses[-1]:.3e}")
        print(f"• Training approach: Automatic adaptation")
        print(f"• Best for: New problems, research, robustness")

    elif choice == 3:
        # Full comparison
        results, analysis = run_full_comparison()

        print(f"\n📊 Full Comparison Complete!")
        print(f"Check the detailed plots and analysis above.")

    elif choice == 4:
        # Quick comparison
        results = compare_soap_methods_quick()

        print(f"\n📊 Quick Comparison Complete!")

    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"=" * 20)
    print(f"Choose SOAP + Fixed Weighting when:")
    print(f"• You know good loss weights for your problem")
    print(f"• Maximum training speed is critical")
    print(f"• You want predictable, reproducible behavior")
    print()
    print(f"Choose SOAP + Gradient Norm Weighting when:")
    print(f"• You're working on new/unknown problems")
    print(f"• You want robust, automatic training")
    print(f"• You're doing research or exploration")
    print(f"• You want to avoid manual weight tuning")

def command_line_interface():
    """Command line interface for the demo"""
    parser = argparse.ArgumentParser(description='SOAP Weighting Demo')
    parser.add_argument('--mode', choices=['fixed', 'grad_norm', 'compare', 'quick'],
                       default='interactive', help='Demo mode')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if args.mode == 'interactive':
        main()
    elif args.mode == 'fixed':
        run_soap_fixed_demo()
    elif args.mode == 'grad_norm':
        run_soap_gradient_norm_demo()
    elif args.mode == 'compare':
        run_full_comparison()
    elif args.mode == 'quick':
        compare_soap_methods_quick()

if __name__ == "__main__":
    # Check if running interactively or with command line args
    import sys
    if len(sys.argv) > 1:
        command_line_interface()
    else:
        main()