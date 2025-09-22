"""
Curriculum Learning Implementation for PINN Training
Based on JAX-PI LDC example with progressive Reynolds number training

This implementation follows the official JAX-PI curriculum strategy:
- Start with low Reynolds number (Re=100)
- Progressively increase to higher Reynolds numbers (Re=400, Re=1000)
- Each stage builds on the previous model weights
- Automatic training step adjustment based on Reynolds number complexity
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import datetime
import uuid
import copy

from settings import Config, PhysicsConfig, TrainingConfig
from training import PINNTrainer
from cavity_flow import CavityFlowProblem


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum learning stage"""
    reynolds_number: float
    max_epochs: int
    stage_name: str

    def __post_init__(self):
        if self.stage_name is None:
            self.stage_name = f"Re_{int(self.reynolds_number)}"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    # Curriculum stages following JAX-PI reference
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(reynolds_number=100.0, max_epochs=5000, stage_name="Re_100"),
        CurriculumStage(reynolds_number=400.0, max_epochs=8000, stage_name="Re_400"),
        CurriculumStage(reynolds_number=1000.0, max_epochs=12000, stage_name="Re_1000")
    ])

    # Transfer learning settings
    transfer_weights: bool = True  # Transfer weights between stages
    reset_optimizer: bool = True   # Reset optimizer state between stages

    # Learning rate scheduling for curriculum
    lr_warmup_epochs: int = 500    # Warmup epochs when transferring to new Re
    lr_warmup_factor: float = 0.1  # Initial LR factor during warmup

    # Convergence criteria for each stage
    convergence_patience: int = 1000  # Epochs to wait for improvement
    convergence_threshold: float = 1e-6  # Minimum loss improvement

    # Adaptive weighting evolution
    reset_weights_between_stages: bool = False  # Keep adaptive weights across stages


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler for Progressive Reynolds Number Training
    Based on JAX-PI LDC implementation strategy
    """

    def __init__(self, base_config: Config, curriculum_config: CurriculumConfig):
        self.base_config = base_config
        self.curriculum_config = curriculum_config
        self.current_stage = 0
        self.total_epochs = 0

        # Training history across all stages
        self.history = {
            'stages': [],
            'reynolds_numbers': [],
            'stage_epochs': [],
            'total_epochs': [],
            'total_loss': [],
            'physics_loss': [],
            'boundary_loss': [],
            'momentum_x': [],
            'momentum_y': [],
            'continuity': [],
            'convergence_flags': [],
            # Adaptive weights tracking
            'weight_ru': [],
            'weight_rv': [],
            'weight_rc': [],
            'weight_u_bc': [],
            'weight_v_bc': []
        }

        # Stage-specific metrics
        self.stage_metrics = {}

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.curriculum_config.stages[self.current_stage]

    def is_complete(self) -> bool:
        """Check if curriculum is complete"""
        return self.current_stage >= len(self.curriculum_config.stages)

    def create_stage_config(self, stage: CurriculumStage) -> Config:
        """Create configuration for specific curriculum stage"""
        # Create deep copy of base config
        stage_config = copy.deepcopy(self.base_config)

        # Update physics parameters for this stage
        stage_config.physics.reynolds_number = stage.reynolds_number
        stage_config.experiment_name = f"{self.base_config.experiment_name}_{stage.stage_name}"

        # Update training parameters
        stage_config.training.epochs = stage.max_epochs

        # Recalculate derived parameters (viscosity) for new Reynolds number
        stage_config._setup_derived_params()

        return stage_config

    def should_advance_stage(self, trainer: PINNTrainer, stage_history: Dict) -> bool:
        """Determine if we should advance to next curriculum stage"""
        if len(stage_history['total_loss']) < self.curriculum_config.convergence_patience:
            return False

        # Check convergence: has loss improved significantly in recent epochs?
        recent_losses = stage_history['total_loss'][-self.curriculum_config.convergence_patience:]
        loss_improvement = max(recent_losses) - min(recent_losses)

        converged = loss_improvement < self.curriculum_config.convergence_threshold

        return converged

    def transfer_weights(self, source_trainer: PINNTrainer, target_trainer: PINNTrainer):
        """Transfer learned weights from previous stage to current stage"""
        if not self.curriculum_config.transfer_weights:
            return

        print(f"  → Transferring weights from previous stage...")

        # Transfer network weights
        target_trainer.network.load_state_dict(source_trainer.network.state_dict())

        # Optionally reset optimizer state for new Reynolds number
        if self.curriculum_config.reset_optimizer:
            # Create fresh optimizer with same config but reset state
            target_trainer._setup_optimizer()
            print(f"  → Optimizer state reset for new Reynolds number")
        else:
            # Transfer optimizer state (not typically recommended for different Re)
            target_trainer.optimizer.load_state_dict(source_trainer.optimizer.state_dict())

    def apply_lr_warmup(self, trainer: PINNTrainer, epoch: int):
        """Apply learning rate warmup when starting new curriculum stage"""
        if epoch < self.curriculum_config.lr_warmup_epochs:
            warmup_factor = self.curriculum_config.lr_warmup_factor + (
                1.0 - self.curriculum_config.lr_warmup_factor
            ) * (epoch / self.curriculum_config.lr_warmup_epochs)

            # Apply warmup factor to current learning rate
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = trainer.base_lr * warmup_factor
        else:
            # Restore normal learning rate schedule
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = trainer.base_lr


class CurriculumTrainer:
    """
    Main Curriculum Learning Trainer
    Implements progressive Reynolds number training following JAX-PI methodology
    """

    def __init__(self, base_config: Config, curriculum_config: Optional[CurriculumConfig] = None):
        self.base_config = base_config
        self.curriculum_config = curriculum_config or CurriculumConfig()
        self.scheduler = CurriculumScheduler(base_config, self.curriculum_config)

        # Runtime tracking
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        self.runtime_id = f"curriculum_{timestamp}_{unique_id}"

        self.results_dir = Path(f"results/curriculum_run_{self.runtime_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track best model across all stages
        self.best_loss = float('inf')
        self.best_model_state = None
        self.best_stage = None

    def train_curriculum(self) -> Tuple[PINNTrainer, Dict]:
        """
        Execute full curriculum learning training

        Returns:
            Tuple of (final_trainer, complete_history)
        """
        print("CURRICULUM LEARNING TRAINING - Progressive Reynolds Number")
        print("=" * 65)
        print("Following JAX-PI methodology:")

        for i, stage in enumerate(self.curriculum_config.stages):
            print(f"  Stage {i+1}: Re={stage.reynolds_number}, max_epochs={stage.max_epochs}")
        print()

        trainer = None

        # Train each curriculum stage
        for stage_idx, stage in enumerate(self.curriculum_config.stages):
            print(f"\n{'='*60}")
            print(f"CURRICULUM STAGE {stage_idx + 1}: {stage.stage_name}")
            print(f"Reynolds Number: {stage.reynolds_number}")
            print(f"Max Epochs: {stage.max_epochs}")
            print(f"{'='*60}")

            # Create configuration for this stage
            stage_config = self.scheduler.create_stage_config(stage)

            # Create new trainer for this stage
            previous_trainer = trainer
            trainer = PINNTrainer(stage_config)

            # Store base learning rate for warmup
            trainer.base_lr = stage_config.training.learning_rate

            # Transfer weights from previous stage if available
            if previous_trainer is not None:
                self.scheduler.transfer_weights(previous_trainer, trainer)

            # Train this stage
            stage_history = self._train_single_stage(trainer, stage, stage_idx)

            # Update scheduler state
            self.scheduler.current_stage = stage_idx + 1

            # Track best model across all stages
            stage_best_loss = min(stage_history['total_loss'])
            if stage_best_loss < self.best_loss:
                self.best_loss = stage_best_loss
                self.best_model_state = trainer.network.state_dict().copy()
                self.best_stage = stage.stage_name

            # Save stage results
            self._save_stage_results(trainer, stage_history, stage, stage_idx)

        # Create final comprehensive results
        self._create_curriculum_analysis()

        return trainer, self.scheduler.history

    def _train_single_stage(self, trainer: PINNTrainer, stage: CurriculumStage, stage_idx: int) -> Dict:
        """Train a single curriculum stage"""

        stage_history = {
            'epochs': [], 'total_loss': [], 'physics_loss': [], 'boundary_loss': [],
            'momentum_x': [], 'momentum_y': [], 'continuity': [],
            'weight_ru': [], 'weight_rv': [], 'weight_rc': [], 'weight_u_bc': [], 'weight_v_bc': []
        }

        print(f"\nTraining Progress (Stage {stage_idx + 1}):")
        print("Epoch | Total Loss | Physics  | Boundary | Weights (ru, rv, rc, u_bc, v_bc)")
        print("-" * 90)

        converged = False

        for epoch in range(stage.max_epochs):
            # Apply learning rate warmup for new Reynolds number
            if epoch < self.curriculum_config.lr_warmup_epochs:
                self.scheduler.apply_lr_warmup(trainer, epoch)

            # Training step
            loss_dict = trainer.train_step()

            # Store metrics
            stage_history['epochs'].append(epoch)
            stage_history['total_loss'].append(loss_dict['total'])
            stage_history['physics_loss'].append(loss_dict.get('physics_total', 0))
            stage_history['boundary_loss'].append(loss_dict.get('boundary_total', 0))
            stage_history['momentum_x'].append(loss_dict['ru'])
            stage_history['momentum_y'].append(loss_dict['rv'])
            stage_history['continuity'].append(loss_dict['rc'])

            # Store adaptive weights
            stage_history['weight_ru'].append(loss_dict.get('weight_ru', 1.0))
            stage_history['weight_rv'].append(loss_dict.get('weight_rv', 1.0))
            stage_history['weight_rc'].append(loss_dict.get('weight_rc', 1.0))
            stage_history['weight_u_bc'].append(loss_dict.get('weight_u_bc', 1.0))
            stage_history['weight_v_bc'].append(loss_dict.get('weight_v_bc', 1.0))

            # Add to global history
            self.scheduler.history['stages'].append(stage.stage_name)
            self.scheduler.history['reynolds_numbers'].append(stage.reynolds_number)
            self.scheduler.history['stage_epochs'].append(epoch)
            self.scheduler.history['total_epochs'].append(self.scheduler.total_epochs + epoch)
            self.scheduler.history['total_loss'].append(loss_dict['total'])
            self.scheduler.history['physics_loss'].append(loss_dict.get('physics_total', 0))
            self.scheduler.history['boundary_loss'].append(loss_dict.get('boundary_total', 0))
            self.scheduler.history['momentum_x'].append(loss_dict['ru'])
            self.scheduler.history['momentum_y'].append(loss_dict['rv'])
            self.scheduler.history['continuity'].append(loss_dict['rc'])
            self.scheduler.history['weight_ru'].append(loss_dict.get('weight_ru', 1.0))
            self.scheduler.history['weight_rv'].append(loss_dict.get('weight_rv', 1.0))
            self.scheduler.history['weight_rc'].append(loss_dict.get('weight_rc', 1.0))
            self.scheduler.history['weight_u_bc'].append(loss_dict.get('weight_u_bc', 1.0))
            self.scheduler.history['weight_v_bc'].append(loss_dict.get('weight_v_bc', 1.0))

            # Log progress
            if epoch % trainer.config.training.log_every == 0:
                weights_str = f"({loss_dict.get('weight_ru', 1.0):.2f}, " \
                             f"{loss_dict.get('weight_rv', 1.0):.2f}, " \
                             f"{loss_dict.get('weight_rc', 1.0):.2f}, " \
                             f"{loss_dict.get('weight_u_bc', 1.0):.2f}, " \
                             f"{loss_dict.get('weight_v_bc', 1.0):.2f})"

                print(f"{epoch:5d} | {loss_dict['total']:8.2e} | "
                      f"{loss_dict['ru']:7.2e} | {loss_dict['u_bc']:7.2e} | {weights_str}")

            # Check convergence
            if self.scheduler.should_advance_stage(trainer, stage_history):
                converged = True
                print(f"\n  → Stage converged at epoch {epoch}")
                break

        # Update total epoch counter
        self.scheduler.total_epochs += len(stage_history['epochs'])

        # Mark convergence in global history
        convergence_flags = [False] * len(stage_history['epochs'])
        if converged:
            convergence_flags[-1] = True
        self.scheduler.history['convergence_flags'].extend(convergence_flags)

        return stage_history

    def _save_stage_results(self, trainer: PINNTrainer, stage_history: Dict, stage: CurriculumStage, stage_idx: int):
        """Save results for individual curriculum stage"""
        stage_dir = self.results_dir / f"stage_{stage_idx+1}_{stage.stage_name}"
        stage_dir.mkdir(exist_ok=True)

        # Save model checkpoint
        model_path = stage_dir / "model_checkpoint.pt"
        torch.save({
            'model_state_dict': trainer.network.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'stage_info': {
                'reynolds_number': stage.reynolds_number,
                'stage_name': stage.stage_name,
                'max_epochs': stage.max_epochs,
                'final_loss': stage_history['total_loss'][-1]
            },
            'stage_history': stage_history
        }, model_path)

        # Create stage visualization
        self._create_stage_visualization(trainer, stage_history, stage, stage_dir)

    def _create_stage_visualization(self, trainer: PINNTrainer, stage_history: Dict, stage: CurriculumStage, stage_dir: Path):
        """Create visualization for individual stage"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        epochs = stage_history['epochs']

        # Loss evolution
        ax1.semilogy(epochs, stage_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss (log scale)')
        ax1.set_title(f'Stage {stage.stage_name}: Loss Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Loss components
        ax2.semilogy(epochs, stage_history['momentum_x'], 'r-', label='X-Momentum', alpha=0.8)
        ax2.semilogy(epochs, stage_history['momentum_y'], 'g-', label='Y-Momentum', alpha=0.8)
        ax2.semilogy(epochs, stage_history['continuity'], 'b-', label='Continuity', alpha=0.8)
        ax2.semilogy(epochs, stage_history['boundary_loss'], 'm-', label='Boundary', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Components (log scale)')
        ax2.set_title('Individual Loss Components')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Adaptive weights evolution
        ax3.plot(epochs, stage_history['weight_ru'], 'r-', label='X-Momentum Weight', linewidth=2)
        ax3.plot(epochs, stage_history['weight_rv'], 'g-', label='Y-Momentum Weight', linewidth=2)
        ax3.plot(epochs, stage_history['weight_rc'], 'b-', label='Continuity Weight', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Adaptive Weight')
        ax3.set_title('Physics Weight Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Solution visualization
        solution = trainer.physics.evaluate_solution(trainer.network, grid_resolution=50)
        speed = solution['speed']
        im = ax4.contourf(solution['x'], solution['y'], speed, levels=20, cmap='viridis')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title(f'Solution: Re={stage.reynolds_number}')
        ax4.set_aspect('equal')
        plt.colorbar(im, ax=ax4, label='Speed')

        plt.tight_layout()

        # Save stage visualization
        fig_path = stage_dir / "stage_results.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  → Stage results saved to: {stage_dir}")

    def _create_curriculum_analysis(self):
        """Create comprehensive curriculum learning analysis"""
        print(f"\n{'='*65}")
        print("CURRICULUM LEARNING ANALYSIS")
        print(f"{'='*65}")

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 15))

        # Create grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        history = self.scheduler.history
        total_epochs = history['total_epochs']
        reynolds_numbers = history['reynolds_numbers']

        # 1. Loss evolution across all stages
        ax1 = fig.add_subplot(gs[0, :])
        ax1.semilogy(total_epochs, history['total_loss'], 'b-', linewidth=2)

        # Add vertical lines for stage transitions
        stage_transitions = []
        current_stage = None
        for i, stage in enumerate(history['stages']):
            if stage != current_stage:
                stage_transitions.append(total_epochs[i])
                current_stage = stage
                if i > 0:  # Don't label the first transition
                    ax1.axvline(total_epochs[i], color='red', linestyle='--', alpha=0.7)
                    ax1.text(total_epochs[i], min(history['total_loss']), f'Re={reynolds_numbers[i]}',
                            rotation=90, verticalalignment='bottom')

        ax1.set_xlabel('Total Epochs')
        ax1.set_ylabel('Total Loss (log scale)')
        ax1.set_title('Curriculum Learning: Complete Training History')
        ax1.grid(True, alpha=0.3)

        # 2. Reynolds number progression
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(total_epochs, reynolds_numbers, 'g-', linewidth=3, marker='o', markersize=2)
        ax2.set_xlabel('Total Epochs')
        ax2.set_ylabel('Reynolds Number')
        ax2.set_title('Reynolds Number Curriculum')
        ax2.grid(True, alpha=0.3)

        # 3. Loss components evolution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.semilogy(total_epochs, history['momentum_x'], 'r-', label='X-Momentum', alpha=0.8)
        ax3.semilogy(total_epochs, history['momentum_y'], 'g-', label='Y-Momentum', alpha=0.8)
        ax3.semilogy(total_epochs, history['continuity'], 'b-', label='Continuity', alpha=0.8)
        ax3.set_xlabel('Total Epochs')
        ax3.set_ylabel('Loss Components (log scale)')
        ax3.set_title('Physics Loss Components')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Adaptive weights evolution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(total_epochs, history['weight_ru'], 'r-', label='X-Momentum', linewidth=2)
        ax4.plot(total_epochs, history['weight_rv'], 'g-', label='Y-Momentum', linewidth=2)
        ax4.plot(total_epochs, history['weight_rc'], 'b-', label='Continuity', linewidth=2)
        ax4.set_xlabel('Total Epochs')
        ax4.set_ylabel('Adaptive Weight')
        ax4.set_title('Weight Evolution Across Stages')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5-7. Final solutions for each stage
        stage_names = list(set(history['stages']))
        final_trainer = PINNTrainer(self.scheduler.create_stage_config(self.curriculum_config.stages[-1]))
        final_trainer.network.load_state_dict(self.best_model_state)

        for i, stage_name in enumerate(stage_names[:3]):  # Show up to 3 stages
            ax = fig.add_subplot(gs[2, i])

            # Find Reynolds number for this stage
            stage_re = None
            for j, s in enumerate(history['stages']):
                if s == stage_name:
                    stage_re = reynolds_numbers[j]
                    break

            # Create config for this Reynolds number
            temp_config = copy.deepcopy(self.base_config)
            temp_config.physics.reynolds_number = stage_re
            temp_config._setup_derived_params()

            # Evaluate solution
            temp_physics = CavityFlowProblem(temp_config.physics, temp_config.training)
            solution = temp_physics.evaluate_solution(final_trainer.network, grid_resolution=50)

            speed = solution['speed']
            im = ax.contourf(solution['x'], solution['y'], speed, levels=20, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Final Solution: {stage_name}')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, label='Speed', shrink=0.8)

        # 8. Performance summary table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')

        # Create performance summary
        summary_text = f"""
CURRICULUM LEARNING PERFORMANCE SUMMARY

Training Stages: {len(self.curriculum_config.stages)}
Total Training Epochs: {max(total_epochs)}
Best Overall Loss: {self.best_loss:.4e} (achieved in {self.best_stage})

Final Loss per Stage:"""

        # Add per-stage statistics
        stage_stats = {}
        for stage in self.curriculum_config.stages:
            stage_indices = [i for i, s in enumerate(history['stages']) if s == stage.stage_name]
            if stage_indices:
                final_loss = history['total_loss'][stage_indices[-1]]
                stage_stats[stage.stage_name] = final_loss
                summary_text += f"\n  {stage.stage_name}: {final_loss:.4e}"

        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Curriculum Learning: Complete Analysis', fontsize=16, fontweight='bold')

        # Save comprehensive analysis
        analysis_path = self.results_dir / "curriculum_analysis.png"
        plt.savefig(analysis_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Save best model
        best_model_path = self.results_dir / "best_model_curriculum.pt"
        torch.save({
            'model_state_dict': self.best_model_state,
            'best_loss': self.best_loss,
            'best_stage': self.best_stage,
            'curriculum_config': self.curriculum_config,
            'complete_history': self.scheduler.history,
            'runtime_id': self.runtime_id
        }, best_model_path)

        print(f"\nCurriculum learning complete!")
        print(f"Best model: {self.best_stage} with loss {self.best_loss:.4e}")
        print(f"Results saved to: {self.results_dir}")
        print(f"Files created:")
        print(f"  - curriculum_analysis.png (comprehensive analysis)")
        print(f"  - best_model_curriculum.pt (best model across all stages)")
        print(f"  - stage_*/ (individual stage results)")


def create_curriculum_config() -> Config:
    """Create configuration optimized for curriculum learning"""
    config = Config()

    # Use SOAP optimizer with gradient norm weighting (best combination)
    config.training.optimizer = "SOAP"
    config.training.learning_rate = 3e-3
    config.training.betas = (0.95, 0.95)
    config.training.weight_decay = 0.01
    config.training.precondition_frequency = 10

    # Enable gradient norm weighting
    config.weighting.scheme = "grad_norm"
    config.weighting.grad_norm = {
        "alpha": 0.9,
        "update_every": 50,
        "eps": 1e-8
    }

    # Adjust sampling for curriculum learning
    config.training.n_collocation = 2000
    config.training.n_boundary = 400
    config.training.log_every = 100

    # Start with low Reynolds number (will be overridden by curriculum)
    config.physics.reynolds_number = 100.0

    return config


def main():
    """Main function demonstrating curriculum learning"""
    print("CURRICULUM LEARNING FOR CAVITY FLOW PINNS")
    print("=" * 50)
    print("Progressive Reynolds Number Training (JAX-PI methodology)")
    print("Stages: Re=100 → Re=400 → Re=1000")
    print()

    # Create base configuration
    base_config = create_curriculum_config()

    # Create curriculum configuration
    curriculum_config = CurriculumConfig(
        stages=[
            CurriculumStage(reynolds_number=100.0, max_epochs=5000, stage_name="Re_100"),
            CurriculumStage(reynolds_number=400.0, max_epochs=8000, stage_name="Re_400"),
            CurriculumStage(reynolds_number=1000.0, max_epochs=12000, stage_name="Re_1000")
        ],
        transfer_weights=True,
        reset_optimizer=True,
        lr_warmup_epochs=500,
        convergence_patience=1000
    )

    # Create and run curriculum trainer
    curriculum_trainer = CurriculumTrainer(base_config, curriculum_config)
    final_trainer, complete_history = curriculum_trainer.train_curriculum()

    print("\nCurriculum Learning Benefits:")
    print("+ Progressive complexity: easier optimization path")
    print("+ Weight transfer: leverages previous learning")
    print("+ Automatic convergence detection")
    print("+ Robust training for high Reynolds numbers")
    print("+ Following proven JAX-PI methodology")

    return curriculum_trainer, final_trainer, complete_history


if __name__ == "__main__":
    # Import dataclass decorator
    from dataclasses import dataclass, field

    # Run curriculum learning demonstration
    curriculum_trainer, trainer, history = main()