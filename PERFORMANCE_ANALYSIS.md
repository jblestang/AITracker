# Tracker Performance Analysis

## Current Implementation Overview

The tracker uses:
- **IMM (Interacting Multiple Model)**: 3 models (CV, CA, CT) with probabilistic switching
- **JPDA (Joint Probabilistic Data Association)**: Handles data association in clutter
- **Kalman Filters**: State estimation for each motion model
- **Track Management**: Initialization, update, and deletion logic

## Performance Metrics

### Available Metrics
1. **Position Error**: Distance between true position and track estimate
2. **Velocity Error**: Difference between true and estimated velocity
3. **Track Coverage**: Percentage of true targets being tracked
4. **Average Missed Detections**: Consecutive missed detections per track
5. **Average Track Age**: How long tracks persist
6. **Existence Probability**: Confidence that track represents a real target
7. **Model Probabilities**: IMM model weights (CV, CA, CT)

## Identified Issues and Improvements

### 1. **Velocity Estimation** ✅ FIXED

**Status:** Implemented improvements to velocity estimation

**Implemented Solutions:**
- ✅ **Improved Initial Velocity Estimation**: 
  - Uses 2-point velocity estimation from previous measurement when available
  - Adaptive velocity uncertainty based on time step (200-500 m²/s²)
  - Enhanced cross-covariance (200-300) scaled with velocity uncertainty
  
- ✅ **Measurement History for Velocity Estimation**:
  - Added `measurement_history` to Track (stores last 3 measurements)
  - Implements least-squares velocity estimation from position history
  - Blends history-based estimate with filter estimate (70% history for young tracks, 30% for mature)
  
- ✅ **Enhanced Cross-Covariance**:
  - Adaptive cross-covariance (200-300) based on velocity uncertainty
  - Maintained after updates to help velocity convergence
  - Scales appropriately with track age and uncertainty

**Remaining Recommendations:**
- Consider innovation-based blending: if innovation is small, trust filter more
- Use velocity consistency checks: if direct estimate and filter estimate differ significantly, investigate

### 2. **IMM Model Probability Adaptation** ✅ FIXED

**Status:** Improved transition probabilities and added covariance inflation

**Implemented Solutions:**
- ✅ **Improved Transition Probabilities**:
  - More adaptive transitions (0.90 stay vs previous 0.95)
  - Prefer transitions between similar models (CV↔CA: 0.05, CA↔CT: 0.05, CV↔CT: 0.025)
  - Better model switching during maneuvers
  
- ✅ **Covariance Inflation During Maneuvers**:
  - Inflates covariance when model probabilities are uncertain (indicating maneuver)
  - Up to 50% inflation based on model probability variance
  - Better uncertainty representation during motion changes

**Remaining Recommendations:**
- **Adaptive Transition Probabilities Based on Motion**:
  - Adjust transition probabilities based on recent motion history
  - Increase probability of staying in current model if it's performing well
  - Use velocity magnitude to inform model transitions (high speed → CV, turning → CT)
  
- **Model-Specific Likelihoods**:
  - Consider computing likelihoods separately for each model using individual measurements
  - Current approach uses weighted measurement which may mask model differences
  - Use maximum likelihood measurement assignment per model for better discrimination

- **Model Probability Smoothing**:
  - Add exponential smoothing to prevent rapid oscillations
  - Use minimum model probability threshold (e.g., 0.05) to prevent models from disappearing

### 3. **JPDA Association Performance** ✅ FIXED

**Status:** Implemented adaptive gates and improved conflict resolution

**Implemented Solutions:**
- ✅ **Adaptive Gate Threshold**:
  - 50% larger gates for tracks < 5 steps old
  - 20% larger gates for tracks < 10 steps old
  - Covariance inflation (1.5x) for young tracks
  - Track-specific gates based on predicted covariance and age
  
- ✅ **Improved Conflict Resolution**:
  - Uses likelihood ratios instead of just distance
  - Considers track consistency (confirmed tracks preferred with 1.2x weight)
  - Better assignment when multiple tracks compete for the same measurement
  - Computes assignment scores using sum of log-likelihoods

**Remaining Recommendations:**
- **Association Quality Metrics**:
  - Track association quality over time
  - Warn when associations are consistently weak
  - Use association history to improve future associations

### 4. **Track Initialization** ✅ FIXED

**Status:** Implemented M/N confirmation logic and improved initialization

**Implemented Solutions:**
- ✅ **M/N Track Confirmation Logic**:
  - Tracks start as tentative (`is_confirmed = false`)
  - Confirmed after M=2 detections in N=3 scans
  - Added `num_detections`, `num_scans` fields to Track
  - Only confirmed tracks used for association (reduces false track initiation)
  
- ✅ **Better Initial Covariance**:
  - Adaptive velocity uncertainty (200-500 m²/s²) based on estimation quality
  - Enhanced cross-covariance (200-300) scaled with velocity uncertainty
  - Position uncertainty from measurement covariance
  
- ✅ **Tentative Track Management**:
  - Tracks created as tentative, promoted to confirmed after M/N criteria
  - Only confirmed tracks used for association in `initialize_new_tracks_from_unassociated`
  - Prevents clutter from creating persistent false tracks

**Remaining Recommendations:**
- Consider different M/N values for different scenarios (M=3, N=5 for high clutter)
- Add track quality history for better deletion decisions

### 5. **Track Deletion Logic** ✅ GOOD

**Current Implementation:**
- Very lenient deletion criteria (20 missed detections, 0.01 existence prob)
- Age-dependent deletion rules
- Good protection for young tracks

**Minor Improvements:**
- Consider adaptive thresholds based on clutter density
- Use track quality history (not just current state) for deletion decisions
- Add "coasting" mode: tracks with low existence prob but good history can persist longer

### 6. **Measurement Noise Modeling** ⚠️ LOW

**Current Issues:**
- Fixed measurement noise (10m std dev)
- No adaptive noise based on measurement quality
- Clutter density is fixed

**Recommendations:**
- **Adaptive Measurement Noise**:
  - Increase noise for measurements far from predicted position
  - Use innovation magnitude to adjust measurement covariance
  - Consider measurement quality indicators if available
  
- **Clutter Modeling**:
  - Current clutter is uniform Poisson
  - Consider spatial clustering of clutter
  - Use track history to identify persistent clutter regions

### 7. **Computational Efficiency** ✅ GOOD

**Current Implementation:**
- Uses `rayon` for parallel processing where possible
- Efficient gating reduces measurement-track pairs
- IMM is computationally efficient

**Potential Optimizations:**
- Cache gate volumes and innovation covariances
- Use approximate JPDA for large numbers of measurements
- Consider track pruning: merge very close tracks

### 8. **State Estimation Accuracy** ✅ PARTIALLY FIXED

**Status:** Implemented covariance inflation during maneuvers

**Implemented Solutions:**
- ✅ **Covariance Inflation During Maneuvers**:
  - Inflates covariance when model probabilities are uncertain (model_prob_variance > 0.15)
  - Up to 50% inflation based on model probability variance
  - Better uncertainty representation during motion changes
  - Computes model uncertainty from probability spread

**Remaining Recommendations:**
- **State Consistency Checks**:
  - Monitor normalized innovation squared (NIS)
  - If NIS consistently high/low, adjust process noise or measurement noise
  - Use chi-squared tests for consistency validation
  
- **Process Noise Inflation**:
  - Add process noise inflation when model probabilities are uncertain
  - Consider adaptive process noise based on maneuver detection

### 9. **Visualization and Monitoring** ✅ GOOD

**Current Features:**
- Real-time tracking statistics
- Error visualization
- Model probability display
- Track quality metrics

**Additional Metrics to Add:**
- **RMS Position Error**: Root mean square error over time
- **RMS Velocity Error**: Velocity estimation accuracy
- **Track Lifecycle**: Track creation, deletion, and duration statistics
- **Association Quality**: Average association probabilities
- **Model Switching Frequency**: How often IMM switches models
- **Innovation Statistics**: Mean and variance of innovations

### 10. **Algorithmic Improvements** ⚠️ ADVANCED

**Potential Enhancements:**
- **Extended Kalman Filter (EKF)**: For nonlinear measurement models (range/azimuth/elevation)
- **Unscented Kalman Filter (UKF)**: Better for highly nonlinear systems
- **Particle Filter**: For non-Gaussian uncertainties
- **Track-to-Track Association**: For multi-sensor scenarios
- **Track Splitting**: Handle closely spaced targets
- **Track Merging**: Combine tracks that converge on same target

## Priority Recommendations

### ✅ Completed (High Priority)
1. ✅ **Improved Initial Velocity Estimation** - Implemented least-squares fit from measurement history
2. ✅ **M/N Track Confirmation** - Implemented M=2/N=3 confirmation logic
3. ✅ **Adaptive Gate Thresholds** - Implemented age-based adaptive gates
4. ✅ **Improved IMM Transition Probabilities** - More adaptive transitions with model preferences
5. ✅ **Enhanced Conflict Resolution** - Uses likelihood ratios and track consistency
6. ✅ **Covariance Inflation During Maneuvers** - Inflates based on model probability uncertainty

### High Priority (Remaining)
1. **Fix Multi-Target Scaling (3+ Aircraft)** ⚠️ - Make track limits dynamic, improve conflict resolution, adaptive thresholds
2. **Add RMS Error Metrics** - Better performance monitoring (position and velocity errors over time)

### Medium Priority (Remaining)
1. **Model-Specific Likelihoods** - Better model discrimination using individual measurements
2. **Association Quality Tracking** - Monitor and improve associations over time
3. **Adaptive Transition Probabilities Based on Motion** - Adjust based on velocity magnitude and motion patterns

### Low Priority (Nice to Have)
1. **Adaptive Measurement Noise** - Handle varying measurement quality
2. **State Consistency Checks** - Automatic tuning
3. **Advanced Filtering** - EKF/UKF for nonlinear models

## Testing Recommendations

1. **Systematic Testing**:
   - Test with different motion patterns (straight, turning, accelerating)
   - Test with varying clutter densities
   - Test with different detection probabilities
   - Test track initialization and deletion scenarios

2. **Performance Benchmarks**:
   - Position error vs. time
   - Velocity error vs. time
   - Track continuity (no false deletions/recreations)
   - Computational time per update

3. **Edge Cases**:
   - Very close targets
   - Targets crossing paths
   - Temporary occlusions
   - High clutter scenarios
   - **3+ targets** (currently has issues with hardcoded limits)

## Conclusion

### Recent Improvements (Completed)
The tracker has been significantly improved with the following fixes:

1. ✅ **Velocity Estimation**: 
   - Least-squares velocity estimation from measurement history
   - Adaptive blending with filter estimates (70% history for young tracks)
   - Enhanced cross-covariance for better velocity convergence

2. ✅ **Track Initialization**: 
   - M/N confirmation logic (M=2, N=3) reduces false track initiation
   - Tentative tracks only promoted after confirmation
   - Only confirmed tracks used for association

3. ✅ **Adaptive Gates**: 
   - Age-based gate thresholds (50% larger for young tracks)
   - Covariance inflation for young tracks
   - Better association for tracks of different maturity levels

4. ✅ **IMM Improvements**: 
   - More adaptive transition probabilities
   - Prefer transitions between similar models
   - Covariance inflation during maneuvers

5. ✅ **Conflict Resolution**: 
   - Uses likelihood ratios instead of distance
   - Considers track consistency (confirmed tracks preferred)

### Issues Identified with 3 Aircraft ⚠️

**Status:** Performance degrades with 3 aircraft due to hardcoded parameters and scaling issues

**Identified Problems:**

1. **Hardcoded Track Limits**:
   - `max_initial_tracks = 2` is hardcoded (should be 3 for 3 aircraft)
   - `expected_num_tracks` defaults to 2 (should match number of simulated aircraft)
   - `max_new_tracks_per_step = 1` may be too restrictive for 3 aircraft
   - Comments still reference "2 aircraft" scenario

2. **Conflict Resolution Scaling**:
   - Conflict resolution works for 2 tracks but may not scale optimally for 3+
   - When 3 tracks compete for the same measurement, the greedy assignment (best track wins) may leave other tracks without good associations
   - No consideration for global assignment optimality (only local per-measurement)

3. **Association Threshold**:
   - 0.5 threshold for unassociated measurements may be too high for 3 aircraft
   - With more tracks and clutter, measurements may have lower individual association probabilities even when correctly associated
   - Threshold should be adaptive based on number of tracks and clutter density

4. **Track Initialization Logic**:
   - Initialization limits prevent creating the 3rd track initially
   - `expected_num_tracks` calculation uses `.max(5)` which is confusing and incorrect
   - Logic assumes 2 aircraft scenario throughout

**Recommended Fixes:**

1. **Make Track Limits Dynamic**:
   - Pass expected number of aircraft from simulation to tracker
   - Set `max_initial_tracks` based on expected aircraft count
   - Set `expected_num_tracks` based on simulation configuration
   - Adjust `max_new_tracks_per_step` based on number of targets

2. **Improve Multi-Track Conflict Resolution**:
   - Consider global assignment optimization (Hungarian algorithm or similar)
   - When multiple tracks compete, consider all possible assignments
   - Use joint probability of all assignments, not just per-measurement greedy

3. **Adaptive Association Thresholds**:
   - Lower threshold when more tracks are present (e.g., 0.3 for 3+ tracks)
   - Consider measurement density and track count
   - Use track-specific thresholds based on track quality

4. **Better Track Initialization**:
   - Initialize all expected tracks from first measurements
   - Use distance-based clustering to identify distinct targets
   - Ensure minimum separation between initial tracks

### Remaining Opportunities
The tracker performs well for single-target and 2-target scenarios with moderate clutter. With 3+ aircraft, performance degrades due to hardcoded parameters. Remaining improvements focus on:
- **Dynamic track limits** based on expected number of targets
- **Improved multi-track conflict resolution** (global assignment optimization)
- **Adaptive association thresholds** based on track count and clutter
- **Performance monitoring** (RMS error metrics)
- **Model-specific likelihoods** for better IMM discrimination
- **Association quality tracking** for adaptive thresholds
- **State consistency checks** (NIS monitoring)

The implemented fixes address the most critical issues for 1-2 targets and should significantly improve tracking accuracy, especially for velocity estimation and false track reduction. However, scaling to 3+ targets requires the additional fixes outlined above.

