#!/usr/bin/env python3
"""
Main training script for PINN with Pirate Networks and SOAP Optimizer
Command-line interface for cavity flow problem
"""

import argparse
import sys
from pathlib import Path
import torch

from settings import Config, default_config
from training import train_cavity_flow, PINNTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train PINN for 2D Cavity Flow")
    
    # Training parameters - use defaults from settings.py
    parser.add_argument("--epochs", type=int, default=default_config.training.epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=default_config.training.learning_rate, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=default_config.training.n_collocation, help="Number of collocation points")
    parser.add_argument("--re", type=float, default=default_config.physics.reynolds_number, help="Reynolds number")
    
    # Optimizer settings - use defaults from settings.py
    parser.add_argument("--optimizer", choices=["SOAP", "Adam"], default=default_config.training.optimizer, help="Optimizer type")
    parser.add_argument("--precond-freq", type=int, default=default_config.training.precondition_frequency, help="SOAP preconditioning frequency")
    
    # Network architecture - use defaults from settings.py
    parser.add_argument("--hidden-dim", type=int, default=default_config.network.hidden_dim, help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=default_config.network.num_layers, help="Number of hidden layers")
    
    # Loss weights
    parser.add_argument("--physics-weight", type=float, default=1.0, help="Physics loss weight")
    parser.add_argument("--boundary-weight", type=float, default=1.0, help="Boundary loss weight")
    
    # Logging and saving
    parser.add_argument("--log-every", type=int, default=100, help="Log frequency")
    parser.add_argument("--save-every", type=int, default=1000, help="Save frequency")
    parser.add_argument("--eval-every", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save-dir", type=str, default="./results", help="Save directory")
    parser.add_argument("--experiment-name", type=str, default="cavity_flow", help="Experiment name")
    
    # Resume training
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    
    return parser.parse_args()

def setup_config(args):
    """Setup configuration from command line arguments"""
    config = Config()
    
    # Update training parameters
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.n_collocation = args.batch_size
    config.training.optimizer = args.optimizer
    config.training.precondition_frequency = args.precond_freq
    config.training.log_every = args.log_every
    config.training.save_every = args.save_every
    config.training.eval_every = args.eval_every
    
    # Update physics parameters
    config.physics.reynolds_number = args.re
    config.physics.physics_weight = args.physics_weight
    config.physics.boundary_weight = args.boundary_weight
    
    # Update network parameters
    config.network.hidden_dim = args.hidden_dim
    config.network.num_layers = args.num_layers
    
    # Setup device
    if args.device == "auto":
        config.network.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config.network.device = args.device
    
    # Setup directories
    config.save_dir = args.save_dir
    config.experiment_name = args.experiment_name
    
    # Recalculate derived parameters
    config._setup_derived_params()
    
    return config

def main():
    """Main training function"""
    args = parse_args()
    
    print("PINN Training for 2D Lid-Driven Cavity Flow")
    print("=" * 50)
    
    # Setup configuration
    config = setup_config(args)
    
    print("Configuration:")
    print(config.summary())
    print()
    
    # Check device availability
    if config.network.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        config.network.device = "cpu"
    
    print(f"Using device: {config.network.device}")
    print()
    
    try:
        # Run training
        trainer = train_cavity_flow(config, resume_from=args.resume)
        
        print("\n" + "=" * 50)
        print("✅ Training completed successfully!")
        print(f"Results saved to: {trainer.save_dir}")
        print(f"Best loss achieved: {trainer.best_loss:.6e}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
