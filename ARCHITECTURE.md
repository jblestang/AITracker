# Architecture Documentation

## System Overview

The AI Tracker system implements a multi-target tracking solution using state-of-the-art algorithms for tracking aircraft in clutter. The system combines:

1. **Interacting Multiple Model (IMM)** for motion model management
2. **Joint Probabilistic Data Association (JPDA)** for data association in clutter
3. **Kalman filtering** for state estimation
4. **3D visualization** for real-time monitoring

## Component Architecture

### 1. State Management (`state.rs`)

**Purpose**: Defines core data structures for the tracking system.

**Key Structures**:
- `State`: Represents target state (position + velocity in 6D space)
- `Measurement`: Sensor measurements (position in 3D space)
- `Track`: Track hypothesis with state, model probabilities, and quality metrics
- `MotionModel`: Enumeration of motion models (CV, CA, CT)

**Design Decisions**:
- 6D state vector: [x, y, z, vx, vy, vz] for 3D tracking
- Direct position measurements (simplified - no range/azimuth conversion)
- Track existence probability for track quality assessment

### 2. Kalman Filters (`kalman.rs`)

**Purpose**: Implements Kalman filters for different motion models.

**Implemented Filters**:
- **CVFilter** (Constant Velocity): Straight-line motion
- **CAFilter** (Constant Acceleration): Accelerating motion
- **CTFilter** (Coordinated Turn): Turning motion (simplified linear approximation)

**Key Operations**:
- `predict()`: State prediction forward in time
- `update()`: State update with measurement
- `likelihood()`: Compute measurement likelihood for association

**Simplifications**:
- CT model uses linear approximation (full implementation would require EKF)
- Process noise models are simplified
- Measurement model assumes direct position measurements

### 3. IMM Algorithm (`imm.rs`)

**Purpose**: Manages multiple motion models probabilistically.

**Algorithm Steps**:
1. **Mixing**: Combine model-conditioned estimates based on transition probabilities
2. **Prediction**: Predict each filter forward
3. **Update**: Update each filter with measurements
4. **Model Probability Update**: Update probabilities based on measurement likelihoods
5. **Combination**: Combine estimates weighted by model probabilities

**Key Parameters**:
- Model transition matrix (Markov chain probabilities)
- Initial model probabilities
- Number of models (typically 3: CV, CA, CT)

**Implementation Details**:
- Uses Bayes' rule for mixing probabilities
- Normalizes probabilities to ensure they sum to 1
- Handles edge cases (zero normalization, etc.)

### 4. JPDA Algorithm (`jpda.rs`)

**Purpose**: Handles data association in clutter environments.

**Key Features**:
- Ellipsoidal gating for computational efficiency
- Association probability computation
- Weighted measurement updates
- Missed detection handling

**Algorithm Steps**:
1. **Gating**: Determine which measurements are in validation gate
2. **Likelihood Computation**: Compute likelihood for each measurement-track pair
3. **Association Probability**: Compute probabilities considering false alarms
4. **Weighted Update**: Update track with weighted combination of measurements

**Key Parameters**:
- Detection probability (Pd): Probability target is detected
- False alarm density (λ_fa): Clutter density per unit volume
- Gate threshold: Chi-squared value for gating

**Simplifications**:
- Simplified association probability computation (not full joint event enumeration)
- Uniform clutter distribution (not realistic for radar)
- Single-sensor only (no multi-sensor fusion)

### 5. Simulation (`simulation.rs`)

**Purpose**: Generates realistic test scenarios.

**Components**:
- `AircraftSimulator`: Generates aircraft trajectories with realistic motion
- `ClutterGenerator`: Generates false alarm measurements
- `Simulation`: Combines both for complete test environment

**Features**:
- Realistic trajectory generation (straight, turns, acceleration)
- Poisson-distributed clutter
- Measurement noise modeling
- Detection probability simulation

**Simplifications**:
- Uniform clutter distribution
- Simplified trajectory phases
- Constant detection probability (not range-dependent)

### 6. Tracker (`tracker.rs`)

**Purpose**: Main tracking system combining IMM and JPDA.

**Key Operations**:
- Track initialization (simplified - from first measurement)
- Track update using IMM-JPDA
- Track deletion (based on missed detections and existence probability)

**Workflow**:
1. For each track:
   - Compute JPDA association probabilities
   - Update IMM with weighted measurements
   - Update track state and quality metrics
2. Delete tracks that should be deleted
3. Initialize new tracks from unassociated measurements (simplified)

**Simplifications**:
- Simple track initialization (not M/N logic)
- Single primary target focus
- Simplified track deletion logic

### 7. Visualization (`visualization.rs`)

**Purpose**: 3D real-time visualization of tracking system.

**Features**:
- True trajectory display
- Track estimate display
- Clutter visualization
- Uncertainty ellipses
- Control panel for simulation control

**Implementation**:
- Uses egui for GUI
- 2D projection of 3D scene (top-down view)
- Real-time updates
- History tracking for trajectory display

## Data Flow

```
Simulation → Measurements (true + clutter)
                ↓
            Tracker
                ↓
        JPDA (Association)
                ↓
        IMM (Model Management)
                ↓
        Kalman Filters (State Estimation)
                ↓
            Track Updates
                ↓
        Visualization
```

## Algorithm Details

### IMM Algorithm

The IMM algorithm maintains multiple Kalman filters, one for each motion model. At each time step:

1. **Mixing**: Computes mixed initial conditions for each filter:
   ```
   μ(i|j) = P(model_i at k | model_j at k+1)
   = transition[i][j] * model_prob[i] / normalization
   ```

2. **Prediction**: Each filter predicts forward independently

3. **Update**: Each filter updates with measurement (if available)

4. **Model Probability Update**:
   ```
   model_prob[j] = likelihood[j] * sum(transition[i][j] * model_prob[i]) / normalization
   ```

5. **Combination**: Weighted combination of model-conditioned estimates

### JPDA Algorithm

JPDA computes association probabilities for all measurement-track pairs:

1. **Gating**: Measurements within validation gate are considered

2. **Association Probability**:
   ```
   β_j = likelihood_j / (sum(likelihoods) + expected_fa + (1-Pd)/Pd)
   β_0 = (1-Pd) / (Pd * normalization)  [missed detection]
   ```

3. **Weighted Update**: Creates pseudo-measurement as weighted sum:
   ```
   z_weighted = sum(β_j * z_j)
   ```

## Performance Considerations

- **Computational Complexity**:
  - IMM: O(M * N) where M = number of models, N = state dimension
  - JPDA: O(T * M) where T = number of tracks, M = number of measurements
  - Gating reduces computational load by filtering measurements

- **Memory Usage**:
  - History limited to 1000 steps for visualization
  - Track state stored per track
  - Model probabilities stored per track

## Testing Strategy

Comprehensive unit tests cover:
- State creation and manipulation
- Kalman filter prediction and update
- IMM mixing and model probability updates
- JPDA association probability computation
- Simulation and clutter generation
- Track initialization and deletion

All tests use realistic scenarios and verify mathematical correctness.

## Future Enhancements

Potential improvements:
1. Extended Kalman Filter (EKF) for nonlinear CT model
2. Full joint event enumeration for JPDA
3. M/N logic for track initialization
4. Multi-sensor fusion
5. Range-dependent detection probability
6. Realistic clutter models
7. Track quality metrics (NNSF, etc.)
8. Multiple target tracking optimization

