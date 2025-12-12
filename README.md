# AI Tracker - Multi-Target Tracking System

A state-of-the-art multi-target tracking system for aircraft tracking in clutter using Interacting Multiple Model (IMM) and Joint Probabilistic Data Association (JPDA) algorithms.

## Architecture

### Core Components

1. **State Estimation**: Kalman filters for different motion models (Constant Velocity, Constant Acceleration, Coordinated Turn)
2. **IMM (Interacting Multiple Model)**: Manages multiple motion models and switches between them probabilistically
3. **JPDA (Joint Probabilistic Data Association)**: Associates measurements to tracks in clutter environments
4. **Simulation**: Generates aircraft trajectories and clutter measurements
5. **Visualization**: 3D real-time visualization using egui

### Algorithm Overview

#### IMM (Interacting Multiple Model)
The IMM algorithm maintains multiple Kalman filters running in parallel, each representing a different motion model. It:
- Mixes model-conditioned estimates based on model transition probabilities
- Updates each filter independently
- Computes model probabilities based on measurement likelihoods
- Combines estimates weighted by model probabilities

#### JPDA (Joint Probabilistic Data Association)
JPDA handles data association in clutter by:
- Computing association probabilities for all measurement-track pairs
- Considering all possible associations simultaneously
- Updating tracks with weighted combinations of measurements
- Handling missed detections and false alarms

## Implementation Status

### Implemented
- ✅ Kalman filter for Constant Velocity (CV) model
- ✅ Kalman filter for Constant Acceleration (CA) model
- ✅ Kalman filter for Coordinated Turn (CT) model
- ✅ IMM algorithm with model switching
- ✅ JPDA data association
- ✅ Aircraft simulation with realistic trajectories
- ✅ Clutter generation (Poisson-distributed false alarms)
- ✅ 3D visualization with egui
- ✅ Comprehensive unit tests

### Simplified/Not Implemented
- ⚠️ **Sensor model**: Simplified to direct position measurements (no range/azimuth conversion)
- ⚠️ **Detection probability**: Constant Pd (not range-dependent)
- ⚠️ **Clutter density**: Uniform spatial distribution (not realistic for radar)
- ⚠️ **Track initialization**: Simple nearest-neighbor initialization (not M/N logic)
- ⚠️ **Track deletion**: Simple deletion based on missed detections (not full track quality metrics)
- ⚠️ **Multiple targets**: Currently optimized for single primary target tracking
- ⚠️ **Sensor fusion**: Single sensor only (no multi-sensor fusion)
- ⚠️ **Gating**: Simple ellipsoidal gating (not optimized for computational efficiency)
- ⚠️ **Model parameters**: Fixed transition probabilities and model parameters (not adaptive)

## Building and Running

```bash
# Build the project
cargo build --release

# Run the application
cargo run --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture
```

## Usage

The application provides a 3D visualization window where you can:
- View aircraft trajectories (true and estimated)
- See clutter measurements
- Observe track estimates
- Control simulation speed and parameters

## Testing

All components have comprehensive unit tests. Run with:
```bash
cargo test
```

Test coverage includes:
- Kalman filter prediction and update
- IMM mixing and model probability updates
- JPDA association probability computation
- Measurement generation and clutter simulation

