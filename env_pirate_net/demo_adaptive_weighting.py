"""
Demonstration script for Gradient Norm Weighting and NTK computation
Shows how to use the new adaptive loss balancing features
"""

import torch
from settings import Config
from training import PINNTrainer

def demo_fixed_weighting():
    """Demonstrate traditional fixed weighting"""
    print("=" * 60)
    print("DEMO 1: Fixed Loss Weighting (Traditional)")
    print("=" * 60)

    config = Config()
    config.weighting.scheme = "fixed"  # Traditional approach
    config.training.epochs = 50
    config.training.log_every = 10

    trainer = PINNTrainer(config)

    print("Training with fixed weights...")
    print("- All loss components have fixed, predefined weights")
    print("- No automatic balancing")
    print()

    # Train for a few epochs
    for epoch in range(5):
        loss_dict = trainer.train_step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch:3d}: Total={loss_dict['total']:.2e} | "
                  f"Physics={loss_dict['ru']:.2e} | "
                  f"Boundary={loss_dict['u_bc']:.2e}")

    print("Fixed weighting completed.\n")


def demo_gradient_norm_weighting():
    """Demonstrate gradient norm weighting"""
    print("=" * 60)
    print("DEMO 2: Gradient Norm Weighting (Auto-Balancing)")
    print("=" * 60)

    config = Config()
    config.weighting.scheme = "grad_norm"
    config.weighting.grad_norm["update_every"] = 5  # Update weights frequently for demo
    config.training.epochs = 50
    config.training.log_every = 10

    trainer = PINNTrainer(config)

    print("Training with gradient norm weighting...")
    print("- Loss weights automatically adjust based on gradient magnitudes")
    print("- Prevents any single loss term from dominating")
    print("- Updates every 5 steps")
    print()

    # Train for more epochs to see weight adaptation
    for epoch in range(20):
        loss_dict = trainer.train_step()
        if epoch % 4 == 0:
            print(f"Epoch {epoch:3d}: Total={loss_dict['total']:.2e} | "
                  f"Physics={loss_dict['ru']:.2e} | "
                  f"Boundary={loss_dict['u_bc']:.2e}")
            print(f"         Weights: ru={loss_dict.get('weight_ru', 1.0):.3f} | "
                  f"u_bc={loss_dict.get('weight_u_bc', 1.0):.3f} | "
                  f"v_bc={loss_dict.get('weight_v_bc', 1.0):.3f}")

    print("Gradient norm weighting completed.\n")


def demo_ntk_weighting():
    """Demonstrate NTK-based weighting"""
    print("=" * 60)
    print("DEMO 3: Neural Tangent Kernel (NTK) Weighting")
    print("=" * 60)

    config = Config()
    config.weighting.scheme = "ntk"
    config.weighting.ntk["update_every"] = 10  # NTK is expensive, update less frequently
    config.training.epochs = 30
    config.training.log_every = 10

    trainer = PINNTrainer(config)

    print("Training with NTK weighting...")
    print("- Loss weights based on Neural Tangent Kernel values")
    print("- Balances optimization dynamics across different loss terms")
    print("- Updates every 10 steps (computationally expensive)")
    print()

    # Train and show NTK-based adaptation
    for epoch in range(15):
        loss_dict = trainer.train_step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Total={loss_dict['total']:.2e} | "
                  f"Physics={loss_dict['ru']:.2e} | "
                  f"Boundary={loss_dict['u_bc']:.2e}")
            print(f"         NTK Weights: ru={loss_dict.get('weight_ru', 1.0):.3f} | "
                  f"u_bc={loss_dict.get('weight_u_bc', 1.0):.3f} | "
                  f"rc={loss_dict.get('weight_rc', 1.0):.3f}")

    print("NTK weighting completed.\n")


def compare_weighting_schemes():
    """Compare different weighting schemes side by side"""
    print("=" * 60)
    print("COMPARISON: All Three Weighting Schemes")
    print("=" * 60)

    schemes = ["fixed", "grad_norm", "ntk"]
    results = {}

    for scheme in schemes:
        print(f"\nTesting {scheme.upper()} weighting...")

        config = Config()
        config.weighting.scheme = scheme
        if scheme == "grad_norm":
            config.weighting.grad_norm["update_every"] = 5
        elif scheme == "ntk":
            config.weighting.ntk["update_every"] = 10

        config.training.epochs = 10
        config.training.log_every = 100  # Quiet

        trainer = PINNTrainer(config)

        # Run 10 training steps
        losses = []
        for epoch in range(10):
            loss_dict = trainer.train_step()
            losses.append(loss_dict['total'])

        results[scheme] = {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'improvement': losses[0] - losses[-1]
        }

        print(f"  Initial loss: {losses[0]:.2e}")
        print(f"  Final loss:   {losses[-1]:.2e}")
        print(f"  Improvement:  {losses[0] - losses[-1]:.2e}")

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    for scheme, result in results.items():
        print(f"{scheme.upper():12s}: "
              f"Initial={result['initial_loss']:.2e} -> "
              f"Final={result['final_loss']:.2e} "
              f"(Δ={result['improvement']:.2e})")

    print("\nKey Insights:")
    print("- Fixed weighting: Simple, but may not balance loss terms optimally")
    print("- Gradient norm: Automatic balancing, prevents gradient pathologies")
    print("- NTK weighting: Theoretically principled, captures training dynamics")
    print()


def show_configuration_options():
    """Show how to configure the adaptive weighting"""
    print("=" * 60)
    print("CONFIGURATION GUIDE")
    print("=" * 60)

    print("1. Fixed Weighting (Traditional):")
    print("   config.weighting.scheme = 'fixed'")
    print("   # Uses weights from config.physics.loss_weights")
    print()

    print("2. Gradient Norm Weighting:")
    print("   config.weighting.scheme = 'grad_norm'")
    print("   config.weighting.grad_norm = {")
    print("       'alpha': 0.9,          # EMA momentum")
    print("       'update_every': 100,   # Update frequency")
    print("       'eps': 1e-8           # Numerical stability")
    print("   }")
    print()

    print("3. Neural Tangent Kernel Weighting:")
    print("   config.weighting.scheme = 'ntk'")
    print("   config.weighting.ntk = {")
    print("       'alpha': 0.9,          # EMA momentum")
    print("       'update_every': 1000,  # Update frequency (expensive!)")
    print("       'eps': 1e-8           # Numerical stability")
    print("   }")
    print()

    print("Tips:")
    print("- Start with gradient norm weighting for most cases")
    print("- Use NTK weighting for research/analysis (slower)")
    print("- Adjust update_every based on computational budget")
    print("- Lower update_every = more frequent adaptation but slower training")


if __name__ == "__main__":
    print("ADAPTIVE LOSS WEIGHTING DEMONSTRATION")
    print("Physics-Informed Neural Networks with Auto-Balancing")
    print()

    # Set random seed for reproducible results
    torch.manual_seed(42)

    try:
        # Show configuration options first
        show_configuration_options()

        # Run demonstrations
        demo_fixed_weighting()
        demo_gradient_norm_weighting()
        demo_ntk_weighting()

        # Compare all schemes
        compare_weighting_schemes()

        print("=" * 60)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("You now have:")
        print("✓ Gradient norm weighting for automatic loss balancing")
        print("✓ Neural Tangent Kernel computation and weighting")
        print("✓ Configuration options for different schemes")
        print("✓ Integration with existing PINN training pipeline")
        print()
        print("These features help prevent gradient pathologies and")
        print("improve PINN training stability and convergence!")

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()