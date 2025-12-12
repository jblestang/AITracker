//! Joint Probabilistic Data Association (JPDA) Algorithm
//! 
//! JPDA handles data association in clutter by computing association probabilities
//! for all measurement-track pairs and updating tracks with weighted combinations
//! of measurements.
//! 
//! Key features:
//! - Handles missed detections
//! - Accounts for false alarms (clutter)
//! - Computes joint association probabilities
//! - Updates tracks with probabilistic associations
//! 
//! Simplified implementation:
//! - Uses ellipsoidal gating for computational efficiency
//! - Assumes Poisson-distributed false alarms
//! - Does not implement full joint event enumeration (uses approximation)

use nalgebra::{Vector3, Matrix3};
use crate::state::{State, Measurement, Track};
use crate::kalman::KalmanFilter;

/// JPDA algorithm implementation
pub struct JPDA {
    /// Detection probability (probability that target is detected)
    pub pd: f64,
    /// False alarm density (clutter density per unit volume)
    pub lambda_fa: f64,
    /// Gate threshold (chi-squared value for gating)
    pub gate_threshold: f64,
}

impl JPDA {
    /// Create a new JPDA filter
    /// 
    /// # Arguments
    /// * `pd` - Detection probability (typically 0.7-0.9)
    /// * `lambda_fa` - False alarm density (clutter per unit volume)
    /// * `gate_threshold` - Chi-squared threshold for gating (typically 9-16)
    pub fn new(pd: f64, lambda_fa: f64, gate_threshold: f64) -> Self {
        Self {
            pd,
            lambda_fa,
            gate_threshold,
        }
    }
    
    /// Create JPDA with default parameters
    pub fn default() -> Self {
        // Use larger gate threshold initially to help with velocity=0 issue
        // 9.0 = 3-sigma, but we need larger for initial tracking
        // lambda_fa should match actual clutter density (1e-8 per m^3)
        // But we need to account for gate volume, so use a value that works with typical gate volumes
        // For a typical gate volume of ~1e6 m^3, lambda_fa * V_g should give reasonable expected_fa
        // With lambda_fa = 1e-8 and V_g = 1e6, expected_fa = 0.01 (very low)
        // But if we use lambda_fa = 1e-8 directly, expected_fa might be too small
        // Use a slightly higher value to account for view-adaptive clutter generation
        Self::new(0.9, 1e-8, 16.0) // pd=0.9, match clutter density 1e-8, 4-sigma gate
    }
    
    /// Compute gate volume for a track
    /// 
    /// The gate is an ellipsoid in measurement space defined by the
    /// innovation covariance. The volume is used to compute the expected
    /// number of false alarms in the gate.
    /// 
    /// # Arguments
    /// * `S` - Innovation covariance matrix
    /// 
    /// # Returns
    /// Gate volume
    fn gate_volume(&self, S: &Matrix3<f64>) -> f64 {
        self.gate_volume_with_threshold(S, self.gate_threshold)
    }
    
    /// Compute gate volume with custom threshold
    fn gate_volume_with_threshold(&self, S: &Matrix3<f64>, threshold: f64) -> f64 {
        // Volume of ellipsoid: V = (4/3) * pi * sqrt(det(S)) * chi^1.5
        // For 3D, simplified: V = (4/3) * pi * sqrt(det(S)) * (threshold)^1.5
        let det_S = S.determinant();
        if det_S > 1e-10 {
            (4.0 / 3.0) * std::f64::consts::PI * det_S.sqrt() * threshold.powf(1.5)
        } else {
            0.0
        }
    }
    
    /// Check if measurement is in gate
    /// 
    /// Uses Mahalanobis distance to determine if measurement is within
    /// the validation gate for a track.
    /// 
    /// # Arguments
    /// * `innovation` - Innovation vector (z - H*x)
    /// * `S` - Innovation covariance matrix
    /// 
    /// # Returns
    /// True if measurement is in gate
    fn in_gate(&self, innovation: &Vector3<f64>, S: &Matrix3<f64>) -> bool {
        self.in_gate_with_threshold(innovation, S, self.gate_threshold)
    }
    
    /// Check if measurement is in gate with custom threshold
    /// 
    /// # Arguments
    /// * `innovation` - Innovation vector (z - H*x)
    /// * `S` - Innovation covariance matrix
    /// * `threshold` - Gate threshold (chi-squared value)
    /// 
    /// # Returns
    /// True if measurement is in gate
    fn in_gate_with_threshold(&self, innovation: &Vector3<f64>, S: &Matrix3<f64>, threshold: f64) -> bool {
        if let Some(S_inv) = S.try_inverse() {
            let d_squared = innovation.transpose() * S_inv * innovation;
            d_squared[0] <= threshold
        } else {
            false
        }
    }
    
    /// Compute association probabilities for a track
    /// 
    /// Computes the probability that each measurement is associated with
    /// the track, and the probability of missed detection.
    /// 
    /// # Arguments
    /// * `track` - Track to associate
    /// * `measurements` - List of measurements
    /// * `filter` - Kalman filter for prediction
    /// * `dt` - Time step
    /// 
    /// # Returns
    /// Vector of association probabilities (one per measurement + missed detection)
    pub fn compute_association_probs(
        &self,
        track: &Track,
        measurements: &[Measurement],
        filter: &dyn KalmanFilter,
        dt: f64,
    ) -> Vec<f64> {
        // Predict track state
        let predicted_state = filter.predict(&track.state, dt);
        
        // Compute innovation covariance
        let H = nalgebra::Matrix3x6::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        );
        // Use default measurement covariance if no measurements
        let default_R = nalgebra::Matrix3::identity() * 100.0;
        let R = measurements.first().map(|m| m.R).unwrap_or(default_R);
        let mut S = H * predicted_state.P * H.transpose() + R;
        
        // Adaptive gate threshold: use larger gates for young tracks
        // Young tracks have more uncertainty, so need larger gates
        let adaptive_gate_threshold = if track.age < 5 {
            self.gate_threshold * 1.5 // 50% larger gate for tracks < 5 steps old
        } else if track.age < 10 {
            self.gate_threshold * 1.2 // 20% larger for tracks < 10 steps old
        } else {
            self.gate_threshold // Normal gate for mature tracks
        };
        
        // Also inflate covariance for young tracks to account for uncertainty
        if track.age < 5 {
            let inflation_factor = 1.5;
            S = S * inflation_factor;
        }
        
        // Compute gate volume with adaptive threshold
        let V_g = self.gate_volume_with_threshold(&S, adaptive_gate_threshold);
        let expected_fa = self.lambda_fa * V_g; // Expected number of false alarms in gate
        
        log::debug!("JPDA compute_association_probs:");
        log::debug!("  Predicted position: ({:.2}, {:.2}, {:.2})", 
            predicted_state.x[0], predicted_state.x[1], predicted_state.x[2]);
        log::debug!("  Predicted velocity: ({:.2}, {:.2}, {:.2})", 
            predicted_state.x[3], predicted_state.x[4], predicted_state.x[5]);
        log::debug!("  Gate volume: {:.2}, Expected FA: {:.4}", V_g, expected_fa);
        log::debug!("  Innovation covariance det: {:.2}", S.determinant());
        log::debug!("  Number of measurements: {}", measurements.len());
        
        // Compute likelihoods for each measurement
        let mut likelihoods = Vec::new();
        let mut in_gate_flags = Vec::new();
        
        for (i, measurement) in measurements.iter().enumerate() {
            let innovation = measurement.z - H * predicted_state.x;
            let innovation_mag = innovation.magnitude();
            // Use adaptive gate threshold
            let in_gate = self.in_gate_with_threshold(&innovation, &S, adaptive_gate_threshold);
            in_gate_flags.push(in_gate);
            
            log::debug!("  Measurement {}: pos=({:.2}, {:.2}, {:.2}), innovation_mag={:.2}, in_gate={}", 
                i, measurement.z[0], measurement.z[1], measurement.z[2], innovation_mag, in_gate);
            
            if in_gate {
                let likelihood = filter.likelihood(&predicted_state, measurement);
                likelihoods.push(likelihood);
                log::debug!("    Likelihood: {:.6}", likelihood);
            } else {
                likelihoods.push(0.0);
                // Check why it's out of gate
                if let Some(S_inv) = S.try_inverse() {
                    let d_squared = innovation.transpose() * S_inv * innovation;
                    log::debug!("    Out of gate: d_squared={:.2}, threshold={:.2}", d_squared[0], self.gate_threshold);
                }
            }
        }
        
        // Count measurements in gate
        let num_in_gate = in_gate_flags.iter().filter(|&&flag| flag).count();
        log::debug!("  Measurements in gate: {}", num_in_gate);
        
        // Compute association probabilities
        // Simplified JPDA: beta_j = likelihood_j / (sum(likelihoods) + expected_fa + (1-pd)/pd)
        let mut association_probs = Vec::new();
        
        if num_in_gate == 0 {
            // No measurements in gate - missed detection
            association_probs.push(1.0);
            for _ in 0..measurements.len() {
                association_probs.push(0.0);
            }
            return association_probs;
        }
        
        // Use log-likelihoods to avoid numerical underflow
        // Standard approach: work in log space, find max, then convert back
        let mut log_likelihoods = Vec::new();
        for &likelihood in &likelihoods {
            if likelihood > 1e-300 {
                log_likelihoods.push(likelihood.ln());
            } else {
                log_likelihoods.push(f64::NEG_INFINITY);
            }
        }
        
        // Find maximum log-likelihood for numerical stability
        let max_log_likelihood = if log_likelihoods.iter().any(|&x| x.is_finite()) {
            log_likelihoods.iter()
                .filter(|&&x| x.is_finite())
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        } else {
            f64::NEG_INFINITY
        };
        
        // If all likelihoods are extremely small (max_log_likelihood < -100),
        // treat as missed detection to avoid numerical issues
        if max_log_likelihood < -100.0 {
            log::debug!("  All likelihoods extremely small (max_log={:.2}), defaulting to missed detection", max_log_likelihood);
            association_probs.push(1.0);
            for _ in 0..measurements.len() {
                association_probs.push(0.0);
            }
            return association_probs;
        }
        
        // Convert back to linear space with numerical stability
        // likelihood_j_normalized = exp(log_likelihood_j - max_log_likelihood)
        let mut normalized_likelihoods = Vec::new();
        for &log_likelihood in &log_likelihoods {
            if log_likelihood.is_finite() {
                normalized_likelihoods.push((log_likelihood - max_log_likelihood).exp());
            } else {
                normalized_likelihoods.push(0.0);
            }
        }
        
        let sum_normalized_likelihoods: f64 = normalized_likelihoods.iter().sum();
        
        // Scale expected_fa and missed term by exp(-max_log_likelihood) to match scale
        // Use a more conservative threshold to avoid numerical issues
        // The key insight: if likelihoods are very small, expected_fa should also be scaled down
        // But we need to be careful not to make expected_fa dominate when it shouldn't
        let scale_factor = if max_log_likelihood > -30.0 {
            (-max_log_likelihood).exp()
        } else {
            // Likelihoods are very small, but not negligible
            // Use a scale that prevents overflow but maintains relative magnitudes
            (-30.0_f64).exp() // Use exp(-30) ≈ 9.36e-14 as a safe scale
        };
        
        // Scale expected_fa: if likelihoods are small, expected_fa should be proportionally small
        // But we also need to account for the fact that expected_fa is already in "per gate" units
        // The issue is that expected_fa might be too large relative to likelihoods
        // For very small likelihoods, we should reduce expected_fa's influence
        let scaled_expected_fa = if max_log_likelihood < -50.0 {
            // If likelihoods are extremely small, reduce expected_fa influence
            expected_fa * scale_factor * 0.1 // Reduce by 90% for very small likelihoods
        } else {
            expected_fa * scale_factor
        };
        
        let scaled_missed_term = if self.pd > 1e-10 {
            (1.0 - self.pd) / self.pd * scale_factor
        } else {
            0.0
        };
        
        let sum_scaled_likelihoods = sum_normalized_likelihoods * scale_factor;
        let normalization = sum_scaled_likelihoods + scaled_expected_fa + scaled_missed_term;
        
        log::debug!("  Max log-likelihood: {:.2}, Sum normalized likelihoods: {:.6e}, Scale factor: {:.6e}", 
            max_log_likelihood, sum_normalized_likelihoods, scale_factor);
        log::debug!("  Normalization: {:.6e} (scaled_likelihoods: {:.6e}, expected_fa: {:.6e}, missed_term: {:.6e})", 
            normalization, sum_scaled_likelihoods, scaled_expected_fa, scaled_missed_term);
        
        // Probability of missed detection: beta_0 = scaled_missed_term / normalization
        let beta_0 = if normalization > 1e-10 {
            scaled_missed_term / normalization
        } else {
            1.0
        };
        association_probs.push(beta_0);
        
        // Association probabilities for each measurement: beta_j = scaled_likelihood_j / normalization
        for (i, &normalized_likelihood) in normalized_likelihoods.iter().enumerate() {
            if normalized_likelihood > 0.0 && normalization > 1e-10 {
                let scaled_likelihood = normalized_likelihood * scale_factor;
                let beta_j = scaled_likelihood / normalization;
                association_probs.push(beta_j);
                log::debug!("  Beta_{} (measurement {}): {:.6e}, normalized_likelihood: {:.6e}", 
                    i+1, i, beta_j, normalized_likelihood);
            } else {
                association_probs.push(0.0);
            }
        }
        
        log::debug!("  Beta_0 (missed detection): {:.6e}", beta_0);
        
        // Normalize to ensure probabilities sum to 1
        let sum: f64 = association_probs.iter().sum();
        log::debug!("  Sum before normalization: {:.6e}", sum);
        if sum > 1e-10 {
            for prob in &mut association_probs {
                *prob /= sum;
            }
        } else {
            // All probabilities are essentially zero - default to missed detection
            log::warn!("  All association probabilities are zero, defaulting to missed detection");
            association_probs[0] = 1.0;
            for i in 1..association_probs.len() {
                association_probs[i] = 0.0;
            }
        }
        log::debug!("  Final association probs: missed={:.4}, measurements={:?}", 
            association_probs[0], 
            association_probs.iter().skip(1).map(|&p| format!("{:.4e}", p)).collect::<Vec<_>>());
        
        association_probs
    }
    
    /// Update track with JPDA associations
    /// 
    /// Updates the track state using weighted combination of measurements
    /// based on association probabilities.
    /// 
    /// # Arguments
    /// * `track` - Track to update
    /// * `measurements` - List of measurements
    /// * `association_probs` - Association probabilities (from compute_association_probs)
    /// * `filter` - Kalman filter
    /// * `dt` - Time step
    /// 
    /// # Returns
    /// Updated track state and whether track was updated (not missed detection)
    pub fn update_track(
        &self,
        track: &Track,
        measurements: &[Measurement],
        association_probs: &[f64],
        filter: &dyn KalmanFilter,
        dt: f64,
    ) -> (State, bool) {
        // Predict state
        let predicted_state = filter.predict(&track.state, dt);
        
        log::debug!("JPDA update_track:");
        log::debug!("  Track state before predict: pos=({:.2}, {:.2}, {:.2}), vel=({:.2}, {:.2}, {:.2})", 
            track.state.x[0], track.state.x[1], track.state.x[2],
            track.state.x[3], track.state.x[4], track.state.x[5]);
        log::debug!("  Predicted state: pos=({:.2}, {:.2}, {:.2}), vel=({:.2}, {:.2}, {:.2})", 
            predicted_state.x[0], predicted_state.x[1], predicted_state.x[2],
            predicted_state.x[3], predicted_state.x[4], predicted_state.x[5]);
        
        // Check if missed detection (first probability is missed detection prob)
        let missed_prob = association_probs[0];
        log::debug!("  Missed detection prob: {:.4}", missed_prob);
        if missed_prob > 0.99 {
            // Missed detection - return predicted state
            log::debug!("  Missed detection - returning predicted state");
            return (predicted_state, false);
        }
        
        // Compute weighted update
        // For JPDA, we compute a pseudo-measurement as weighted sum of measurements
        
        // Weighted measurement
        let mut weighted_z = Vector3::zeros();
        let mut total_weight = 0.0;
        
        for (i, measurement) in measurements.iter().enumerate() {
            let beta = association_probs[i + 1]; // +1 because first is missed detection
            if beta > 1e-6 {
                weighted_z += beta * measurement.z;
                total_weight += beta;
            }
        }
        
        if total_weight > 1e-6 {
            weighted_z /= total_weight;
            
            // Create pseudo-measurement with adjusted covariance
            // Use a more conservative covariance adjustment for stability
            // The covariance should account for the spread of measurements
            let base_r = measurements[0].R;
            // Adjust covariance: larger when associations are uncertain (low total_weight)
            let r_adjustment = (1.0 / total_weight).min(10.0); // Cap adjustment to prevent instability
            let r = base_r * r_adjustment;
            let pseudo_measurement = Measurement::new(weighted_z, r, measurements[0].time);
            
            // Update with pseudo-measurement
            let (updated_state, _) = filter.update(&predicted_state, &pseudo_measurement);
            
            log::debug!("  Updated state: pos=({:.2}, {:.2}, {:.2}), vel=({:.2}, {:.2}, {:.2})", 
                updated_state.x[0], updated_state.x[1], updated_state.x[2],
                updated_state.x[3], updated_state.x[4], updated_state.x[5]);
            
            (updated_state, true)
        } else {
            // No valid associations - return predicted state
            (predicted_state, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kalman::CVFilter;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_jpda_creation() {
        let jpda = JPDA::default();
        assert_eq!(jpda.pd, 0.9);
        assert!(jpda.lambda_fa > 0.0);
    }
    
    #[test]
    fn test_gate_volume() {
        let jpda = JPDA::default();
        let S = nalgebra::Matrix3::identity() * 100.0;
        let volume = jpda.gate_volume(&S);
        
        assert!(volume > 0.0);
    }
    
    #[test]
    fn test_in_gate() {
        let jpda = JPDA::default();
        let S = nalgebra::Matrix3::identity() * 100.0;
        
        // Measurement at origin (should be in gate)
        let innovation = Vector3::zeros();
        assert!(jpda.in_gate(&innovation, &S));
        
        // Measurement far away (should be out of gate)
        let far_innovation = Vector3::new(1000.0, 1000.0, 1000.0);
        assert!(!jpda.in_gate(&far_innovation, &S));
    }
    
    #[test]
    fn test_association_probs_no_measurements() {
        let jpda = JPDA::default();
        let filter = CVFilter::new(1.0);
        let state = crate::state::State::zero();
        let track = Track::new(0, state, 3, 0.0);
        let measurements = vec![];
        
        let probs = jpda.compute_association_probs(&track, &measurements, &filter, 1.0);
        
        // Should have missed detection probability = 1.0
        assert_eq!(probs.len(), 1);
        assert_relative_eq!(probs[0], 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_association_probs_with_measurements() {
        let jpda = JPDA::default();
        let filter = CVFilter::new(1.0);
        let x = nalgebra::Vector6::new(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let P = nalgebra::Matrix6::identity() * 10.0;
        let state = crate::state::State::new(x, P);
        let track = Track::new(0, state, 3, 0.0);
        
        let measurement = Measurement::new(
            Vector3::new(110.0, 210.0, 310.0),
            nalgebra::Matrix3::identity() * 10.0,
            1.0,
        );
        let measurements = vec![measurement];
        
        let probs = jpda.compute_association_probs(&track, &measurements, &filter, 1.0);
        
        // Should have missed detection prob + one measurement prob
        assert_eq!(probs.len(), 2);
        
        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_update_track_missed_detection() {
        let jpda = JPDA::default();
        let filter = CVFilter::new(1.0);
        let x = nalgebra::Vector6::new(100.0, 200.0, 300.0, 10.0, 20.0, 30.0);
        let P = nalgebra::Matrix6::identity() * 10.0;
        let state = crate::state::State::new(x, P);
        let track = Track::new(0, state, 3, 0.0);
        
        let measurements = vec![];
        let association_probs = vec![1.0]; // Missed detection
        
        let (updated_state, was_updated) = jpda.update_track(
            &track,
            &measurements,
            &association_probs,
            &filter,
            1.0,
        );
        
        assert!(!was_updated);
        // Should be predicted state (moved forward)
        assert!(updated_state.x[0] > track.state.x[0]);
    }
    
    #[test]
    fn test_update_track_with_measurement() {
        let jpda = JPDA::default();
        let filter = CVFilter::new(1.0);
        let x = nalgebra::Vector6::new(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let P = nalgebra::Matrix6::identity() * 10.0;
        let state = crate::state::State::new(x, P);
        let track = Track::new(0, state, 3, 0.0);
        
        let measurement = Measurement::new(
            Vector3::new(110.0, 210.0, 310.0),
            nalgebra::Matrix3::identity() * 10.0,
            1.0,
        );
        let measurements = vec![measurement];
        let association_probs = vec![0.1, 0.9]; // 10% missed, 90% association
        
        let (updated_state, was_updated) = jpda.update_track(
            &track,
            &measurements,
            &association_probs,
            &filter,
            1.0,
        );
        
        assert!(was_updated);
        // Updated state should be between prediction and measurement
        assert!(updated_state.x[0] > 100.0);
    }
}

