# Algorithm Documentation

## Overview

This document provides detailed mathematical descriptions of the tracking algorithms implemented in the AI Tracker system.

## 1. Kalman Filter

### State Space Model

The Kalman filter assumes a linear state space model:

**State Equation**:
```
x(k+1) = F(k) * x(k) + w(k)
```

**Measurement Equation**:
```
z(k) = H(k) * x(k) + v(k)
```

Where:
- `x(k)` is the state vector (6D: position + velocity)
- `F(k)` is the state transition matrix
- `w(k)` is process noise (zero-mean Gaussian, covariance Q)
- `z(k)` is the measurement vector (3D: position)
- `H(k)` is the measurement matrix
- `v(k)` is measurement noise (zero-mean Gaussian, covariance R)

### Prediction Step

**State Prediction**:
```
x_pred(k+1|k) = F(k) * x(k|k)
```

**Covariance Prediction**:
```
P_pred(k+1|k) = F(k) * P(k|k) * F'(k) + Q(k)
```

### Update Step

**Innovation**:
```
y(k+1) = z(k+1) - H(k+1) * x_pred(k+1|k)
```

**Innovation Covariance**:
```
S(k+1) = H(k+1) * P_pred(k+1|k) * H'(k+1) + R(k+1)
```

**Kalman Gain**:
```
K(k+1) = P_pred(k+1|k) * H'(k+1) * S^(-1)(k+1)
```

**State Update**:
```
x(k+1|k+1) = x_pred(k+1|k) + K(k+1) * y(k+1)
```

**Covariance Update**:
```
P(k+1|k+1) = (I - K(k+1) * H(k+1)) * P_pred(k+1|k)
```

### Motion Models

#### Constant Velocity (CV)

State transition matrix:
```
F = [I_3  dt*I_3]
    [0_3  I_3   ]
```

Where `I_3` is 3x3 identity, `0_3` is 3x3 zero matrix, `dt` is time step.

#### Constant Acceleration (CA)

Similar to CV but with larger process noise to model acceleration uncertainty.

#### Coordinated Turn (CT)

Simplified linear approximation:
```
F = [I_3  dt*I_3        ]
    [0_3  I_3 + dt*Ω   ]
```

Where `Ω` is turn rate coupling matrix (simplified).

## 2. Interacting Multiple Model (IMM)

### Algorithm Overview

IMM maintains M parallel Kalman filters, one for each motion model. It probabilistically switches between models based on measurement likelihoods.

### Step 1: Mixing

Compute mixing probabilities using Bayes' rule:
```
μ(i|j) = P(model_i at k | model_j at k+1)
       = π(i,j) * μ_i(k) / c_j
```

Where:
- `π(i,j)` is the model transition probability
- `μ_i(k)` is the model probability at time k
- `c_j = sum_i(π(i,j) * μ_i(k))` is normalization constant

Mixed initial conditions:
```
x_0j = sum_i(μ(i|j) * x_i(k|k))
P_0j = sum_i(μ(i|j) * [P_i(k|k) + (x_i(k|k) - x_0j) * (x_i(k|k) - x_0j)'])
```

### Step 2: Prediction

Each filter predicts forward:
```
x_j(k+1|k) = F_j * x_0j
P_j(k+1|k) = F_j * P_0j * F_j' + Q_j
```

### Step 3: Update

Each filter updates with measurement:
```
x_j(k+1|k+1) = x_j(k+1|k) + K_j * y_j
P_j(k+1|k+1) = (I - K_j * H_j) * P_j(k+1|k)
```

### Step 4: Model Probability Update

Update model probabilities:
```
Λ_j = likelihood_j = N(y_j; 0, S_j)
c = sum_j(Λ_j * sum_i(π(i,j) * μ_i(k)))
μ_j(k+1) = Λ_j * sum_i(π(i,j) * μ_i(k)) / c
```

Where `N(y; 0, S)` is the multivariate Gaussian probability density.

### Step 5: Combination

Combine model-conditioned estimates:
```
x(k+1|k+1) = sum_j(μ_j(k+1) * x_j(k+1|k+1))
P(k+1|k+1) = sum_j(μ_j(k+1) * [P_j(k+1|k+1) + (x_j - x) * (x_j - x)'])
```

## 3. Joint Probabilistic Data Association (JPDA)

### Problem Formulation

Given:
- T tracks
- M measurements
- Clutter (false alarms)

Goal: Associate measurements to tracks probabilistically.

### Gating

Ellipsoidal validation gate:
```
(y - H*x_pred)' * S^(-1) * (y - H*x_pred) ≤ γ
```

Where `γ` is the gate threshold (chi-squared value).

### Association Probabilities

For each track, compute association probabilities:

**Normalization Constant**:
```
c = sum_j(likelihood_j) + λ_fa * V_g + (1-Pd)/Pd
```

Where:
- `likelihood_j` is measurement likelihood
- `λ_fa` is false alarm density
- `V_g` is gate volume
- `Pd` is detection probability

**Association Probabilities**:
```
β_j = likelihood_j / c  (for measurement j)
β_0 = (1-Pd) / (Pd * c)  (missed detection)
```

### Weighted Update

Create pseudo-measurement:
```
z_weighted = sum_j(β_j * z_j)
R_weighted = R / sum_j(β_j)
```

Update track with pseudo-measurement using standard Kalman update.

### Gate Volume

For 3D ellipsoid:
```
V_g = (4/3) * π * sqrt(det(S)) * γ^1.5
```

## 4. Track Management

### Track Initialization

Simplified initialization from first measurement:
- Initialize position from measurement
- Initialize velocity to zero with large uncertainty
- Initialize model probabilities uniformly

**Note**: Full implementation would use M/N logic (M detections out of N scans).

### Track Deletion

Track is deleted if:
- Missed detections ≥ max_missed_detections
- Existence probability < min_existence_probability

### Track Quality

Track quality metrics:
- **Age**: Number of updates
- **Missed Detections**: Consecutive missed detections
- **Existence Probability**: Probability track represents real target

## Mathematical Properties

### Consistency

Kalman filter maintains consistency:
- E[x_est] = E[x_true]
- P_est ≥ P_true (covariance bounds true uncertainty)

### Optimality

Under linear-Gaussian assumptions:
- Kalman filter is optimal (minimum mean square error)
- IMM is suboptimal but handles model uncertainty
- JPDA is suboptimal but handles data association uncertainty

### Computational Complexity

- Kalman filter: O(N³) where N is state dimension
- IMM: O(M * N³) where M is number of models
- JPDA: O(T * M * N³) where T is tracks, M is measurements
- Gating reduces M significantly

## References

1. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004). *Estimation with applications to tracking and navigation*. John Wiley & Sons.

2. Bar-Shalom, Y., & Fortmann, T. E. (1988). *Tracking and data association*. Academic Press.

3. Blackman, S., & Popoli, R. (1999). *Design and analysis of modern tracking systems*. Artech House.

4. Li, X. R., & Jilkov, V. P. (2003). Survey of maneuvering target tracking. Part I: Dynamic models. *IEEE Transactions on Aerospace and Electronic Systems*, 39(4), 1333-1364.

