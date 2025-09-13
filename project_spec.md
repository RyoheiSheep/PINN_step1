# Project Specification

## Project Overview

This project develops a Physics-Informed Neural Network (PINN) system combining the Pirate network architecture with the SOAP optimizer for solving 2D incompressible fluid flow problems. The implementation focuses on the lid-driven cavity flow problem as a benchmark test case.

**Key Innovation**: Converting the JAX-based Pirate network to PyTorch while integrating the second-order SOAP optimizer for enhanced training stability and convergence.

## Objectives

- [ ] **Primary**: Implement Pirate network architecture in PyTorch for physics-informed neural networks
- [ ] **Secondary**: Integrate SOAP optimizer for second-order optimization in PINN training
- [ ] **Application**: Solve 2D incompressible lid-driven cavity flow problem
- [ ] **Validation**: Demonstrate convergence to physically accurate flow solutions

## Scope
### In Scope

- **Network Architecture**: Pirate network implementation in PyTorch
- **Optimization**: SOAP optimizer integration for training
- **Physics Problem**: 2D incompressible lid-driven cavity flow
- **Data Generation**: Collocation points (interior) and boundary points sampling
- **Loss Functions**: Navier-Stokes physics loss + boundary condition loss
- **Visualization**: Flow field plots (velocity vectors, streamlines, pressure contours)

### Out of Scope

- **Scalability**: Multi-GPU distributed training
- **Other Physics**: Heat transfer, wave equations, or other PDE problems
- **Advanced Features**: Real-time visualization, interactive plots
- **Optimization**: Automated hyperparameter tuning or grid search
- **Benchmarking**: Comparative studies with other optimizers (Adam, L-BFGS, etc.)


## Technical Requirements
### Functional Requirements

1. **Configuration Management**: Network architecture and training parameters configurable via `settings.py`
2. **Framework Conversion**: PyTorch implementation adapted from JAX-based Pirate network
3. **Optimization**: SOAP optimizer integration with proper gradient handling
4. **Physics Modeling**: Navier-Stokes equation residuals computation for 2D incompressible flow
5. **Boundary Conditions**: Cavity flow constraints (no-slip walls, moving lid)
6. **Post-Processing**: Model inference and flow field visualization capabilities

### Non-Functional Requirements

- **Performance**: Target combined loss (physics + boundary) approaching zero over fixed training epochs
- **Hardware**: CUDA-compatible GPU required for efficient training
- **Software Stack**: Python 3.12.3, PyTorch (latest stable), NumPy, Matplotlib
- **Usability**: Intuitive configuration through centralized settings file
- **Code Quality**: Compact, problem-specific implementation following KISS principle
- **Maintainability**: Clear code structure with minimal external dependencies 

## Architecture Overview
<!-- High-level system architecture -->
```
Settings.py → Network Config + Training Settings
     ↓
Pirate Network (PyTorch)
     ↓
Collocation Points → Forward Pass → Physics Loss (Navier-Stokes)
Boundary Points → Forward Pass → Boundary Loss (Cavity conditions)
     ↓
Total Loss = Physics Loss + Boundary Loss
     ↓
SOAP Optimizer → Backpropagation
     ↓
Trained Model → Inference → Visualization
```

**Key Components:**
- **Settings.py**: Network architecture + training parameters (epochs, learning rate, point counts)
- **Pirate Network**: Residual adaptive network architecture in PyTorch
- **Physics Loss**: 2D incompressible Navier-Stokes equation residuals
- **Boundary Loss**: Cavity flow boundary conditions (no-slip walls, lid velocity)
- **SOAP Optimizer**: Second-order adaptive optimizer for training
- **Data Generation**: Collocation points in domain, boundary points on walls

## Resources Required

**Hardware**:
- CUDA-compatible GPU (minimum 4GB VRAM recommended)
- CPU with sufficient RAM for data preprocessing

**Software Environment**:
- Python 3.12.3
- PyTorch (latest stable version)  
- CUDA toolkit compatible with PyTorch
- Development tools: Git, text editor/IDE

## Implementation Plan

### Phase 1: Network Architecture Implementation
**Goal**: Convert Pirate network from JAX to PyTorch
**Deliverables**:
- PyTorch implementation of Pirate network architecture
- Configuration system via `settings.py` 
- Basic network forward pass functionality
- Unit tests for network components

### Phase 2: Physics Problem Setup  
**Goal**: Implement cavity flow physics and data generation
**Deliverables**:
- Collocation point sampling within domain
- Boundary point generation for cavity walls
- Navier-Stokes physics loss computation
- Boundary condition loss functions

### Phase 3: Training System Integration
**Goal**: Complete training pipeline with SOAP optimizer
**Deliverables**:
- SOAP optimizer integration 
- Training loop with loss monitoring
- Gradient computation and backpropagation
- Training progress logging and checkpointing

### Phase 4: Visualization and Validation
**Goal**: Post-processing and result analysis
**Deliverables**:
- Model inference for trained networks
- Flow field visualization (velocity, pressure, streamlines)
- Solution validation against expected physics
- Documentation and usage examples

## Success Criteria

**Technical Implementation**:
- ✅ Pirate network successfully implemented in PyTorch
- ✅ SOAP optimizer integrates without errors
- ✅ Configuration system works correctly

**Training Performance**:
- ✅ Combined loss (physics + boundary) decreases during training
- ✅ Target loss approaches zero (< 1e-4 for well-conditioned problems)
- ✅ Training completes without numerical instabilities

**Physical Accuracy**:
- ✅ Flow field exhibits expected cavity flow patterns
- ✅ Boundary conditions properly satisfied (no-slip walls, lid motion)
- ✅ Mass conservation approximately satisfied (∇·u ≈ 0)

**System Quality**:
- ✅ Code executes end-to-end without errors
- ✅ Visualization produces clear, interpretable plots  
- ✅ Settings.py enables easy configuration changes

## Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1**: Network Architecture | 2-3 days | Pirate network in PyTorch, settings.py |
| **Phase 2**: Physics Implementation | 2-3 days | Data generation, loss functions |
| **Phase 3**: Training Integration | 2-3 days | SOAP optimizer, training loop |
| **Phase 4**: Visualization & Testing | 1-2 days | Inference, plots, validation |
| **Total Project Duration** | **7-11 days** | **Complete PINN system** |





## Dependencies

**Internal Resources**:
- JAX-based Pirate network implementation (in `pirate_net/` directory)
- SOAP optimizer PyTorch implementation (in `SOAP/` directory)
- Reference examples and configurations from existing codebase

**External Dependencies**:
- PyTorch (latest stable version)
- NumPy (for numerical computations)
- Matplotlib (for visualization)
- CUDA toolkit (for GPU acceleration) 

## Assumptions

- **Architecture Source**: Pirate network architecture can be successfully converted from JAX to PyTorch
- **Optimizer Compatibility**: SOAP optimizer integrates well with PyTorch autograd system
- **Hardware Availability**: CUDA-compatible GPU available for training
- **Problem Complexity**: 2D cavity flow is sufficiently representative for validation
- **Reference Code**: Existing implementations provide sufficient guidance for conversion

## Development Guidelines

**Core Principle**: Keep It Simple, Stupid (KISS)
- Implement specifically for 2D cavity flow problem only
- Avoid over-engineering or excessive generalization  
- Prioritize compact, readable code over complex abstractions
- Focus on working solution rather than framework development

**Code Organization**:
- Single-purpose modules for network, training, and visualization
- Minimal external dependencies beyond PyTorch ecosystem
- Clear separation between physics, network, and optimization components
