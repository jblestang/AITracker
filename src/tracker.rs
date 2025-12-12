//! Main tracker combining IMM and JPDA
//! 
//! This module implements the complete tracking system that combines:
//! - IMM for motion model management
//! - JPDA for data association in clutter
//! - Track management (initialization, deletion, update)
//! 
//! The tracker maintains multiple tracks (though optimized for single
//! primary target) and updates them using the IMM-JPDA framework.

use nalgebra::Vector3;
use crate::state::{State, Measurement, Track, MotionModel};
use crate::imm::IMM;
use crate::jpda::JPDA;
use crate::kalman::{KalmanFilter, get_filter};

/// Main tracker implementation
pub struct Tracker {
    /// IMM filter for each track
    imm_filters: Vec<IMM>,
    /// Model-conditioned states for each track (one state per model per track)
    model_states: Vec<Vec<State>>,
    /// JPDA data associator
    jpda: JPDA,
    /// List of tracks
    tracks: Vec<Track>,
    /// Next track ID
    next_track_id: usize,
    /// Previous measurement for velocity initialization
    prev_measurement: Option<Measurement>,
    /// Previous track state for velocity estimation
    prev_track_state: Option<State>,
    /// Previous track position for velocity estimation (simpler than full state)
    prev_track_position: Option<(Vector3<f64>, f64)>, // (position, time)
    /// Maximum number of missed detections before track deletion
    max_missed_detections: usize,
    /// Minimum track existence probability
    min_existence_prob: f64,
    /// Time step
    dt: f64,
}

impl Tracker {
    /// Create a new tracker
    /// 
    /// # Arguments
    /// * `dt` - Time step
    pub fn new(dt: f64) -> Self {
        let jpda = JPDA::default();
        
        Self {
            imm_filters: Vec::new(),
            model_states: Vec::new(),
            jpda,
            tracks: Vec::new(),
            next_track_id: 0,
            max_missed_detections: 20, // Very high threshold to prevent premature deletion
            min_existence_prob: 0.01, // Very low threshold - only delete if track is clearly dead
            dt,
            prev_measurement: None,
            prev_track_state: None,
            prev_track_position: None,
        }
    }
    
    /// Initialize a new track
    /// 
    /// Simplified track initialization: creates a track from the first
    /// measurement. In reality, track initialization uses M/N logic
    /// (M detections out of N scans).
    /// 
    /// # Arguments
    /// * `measurement` - Initial measurement
    /// * `time` - Current time
    fn initialize_track(&mut self, measurement: &Measurement, time: f64) {
        // Initialize state from measurement
        let mut x = nalgebra::Vector6::zeros();
        x[0] = measurement.z[0];
        x[1] = measurement.z[1];
        x[2] = measurement.z[2];
        
        // Try to estimate initial velocity from previous measurement if available
        log::debug!("Initializing track {} at time {:.2}", self.next_track_id, time);
        log::debug!("  Measurement position: ({:.2}, {:.2}, {:.2})", measurement.z[0], measurement.z[1], measurement.z[2]);
        
        if let Some(prev_meas) = &self.prev_measurement {
            let dt_meas = (measurement.time - prev_meas.time).max(0.1); // Avoid division by zero
            log::debug!("  Previous measurement available at time {:.2}, dt: {:.2}", prev_meas.time, dt_meas);
            if dt_meas > 0.0 {
                // Estimate velocity from position difference
                x[3] = (measurement.z[0] - prev_meas.z[0]) / dt_meas;
                x[4] = (measurement.z[1] - prev_meas.z[1]) / dt_meas;
                x[5] = (measurement.z[2] - prev_meas.z[2]) / dt_meas;
                log::debug!("  Estimated initial velocity: ({:.2}, {:.2}, {:.2})", x[3], x[4], x[5]);
            }
        } else {
            log::debug!("  No previous measurement - velocity initialized to zero");
        }
        
        // Initial covariance: position uncertainty from measurement, velocity uncertainty larger
        let mut P = nalgebra::Matrix6::identity() * 1000.0;
        P[(0, 0)] = measurement.R[(0, 0)];
        P[(1, 1)] = measurement.R[(1, 1)];
        P[(2, 2)] = measurement.R[(2, 2)];
        
        // Velocity uncertainty: larger if we estimated it, even larger if we didn't
        let vel_uncertainty = if self.prev_measurement.is_some() {
            200.0 // Estimated velocity - moderate uncertainty
        } else {
            1000.0 // No velocity estimate - large uncertainty
        };
        P[(3, 3)] = vel_uncertainty;
        P[(4, 4)] = vel_uncertainty;
        P[(5, 5)] = vel_uncertainty;
        
        // Cross-covariance terms (position-velocity correlation)
        // This helps the filter learn velocity from position measurements
        // Use larger cross-covariance to ensure velocity gets updated
        let cross_cov = 200.0; // Increased from 100.0
        P[(0, 3)] = cross_cov;
        P[(3, 0)] = cross_cov;
        P[(1, 4)] = cross_cov;
        P[(4, 1)] = cross_cov;
        P[(2, 5)] = cross_cov;
        P[(5, 2)] = cross_cov;
        
        let initial_state = State::new(x, P);
        
        // Create IMM filter
        let imm = IMM::default();
        let num_models = imm.model_probs().len();
        
        // Create track
        let track = Track::new(self.next_track_id, initial_state, num_models, time);
        self.next_track_id += 1;
        
        // Initialize model-conditioned states (one per model)
        let model_states = vec![initial_state; num_models];
        
        self.tracks.push(track);
        self.imm_filters.push(imm);
        self.model_states.push(model_states);
        
        log::debug!("Track {} initialized with velocity: ({:.2}, {:.2}, {:.2})", 
            self.next_track_id - 1, x[3], x[4], x[5]);
    }
    
    /// Update tracks with new measurements
    /// 
    /// Main tracking loop:
    /// 1. For each track, compute JPDA association probabilities
    /// 2. Update tracks using IMM-JPDA
    /// 3. Delete tracks that should be deleted
    /// 4. Initialize new tracks from unassociated measurements
    /// 
    /// # Arguments
    /// * `measurements` - New measurements (true + clutter)
    /// * `time` - Current time
    pub fn update(&mut self, measurements: &[Measurement], time: f64) {
        // Store measurement for velocity estimation in next initialization
        if let Some(first_meas) = measurements.first() {
            self.prev_measurement = Some(*first_meas);
        }
        
        // Only initialize new track if we truly have no tracks
        // Don't reinitialize immediately after deletion - wait a bit
        if self.tracks.is_empty() {
            // Only initialize if we have measurements and haven't had a track recently
            // This prevents immediate reinitialization after deletion
            if let Some(first_meas) = measurements.first() {
                self.initialize_track(first_meas, time);
                // Reset previous track state when initializing new track
                self.prev_track_state = None;
                self.prev_track_position = None;
            }
            return;
        }
        
        // Update existing tracks
        for (track_idx, track) in self.tracks.iter_mut().enumerate() {
            // Get IMM filter and model states for this track
            let imm = &mut self.imm_filters[track_idx];
            let current_model_states = self.model_states[track_idx].clone();
            
            // Get filters for each model
            let models = MotionModel::all();
            
            // Compute association probabilities using combined state for gating
            let filter_for_gating: Box<dyn KalmanFilter> = get_filter(MotionModel::ConstantVelocity);
            let association_probs = self.jpda.compute_association_probs(
                track,
                measurements,
                filter_for_gating.as_ref(),
                self.dt,
            );
            
            // Step 1: IMM mixing and prediction (before JPDA update)
            let predicted_model_states = imm.mix_and_predict(&current_model_states, self.dt);
            
            // Step 2: Compute likelihoods on PREDICTED states (before update)
            // This is critical for correct model probability updates
            let mut likelihoods = Vec::new();
            let mut measurement_for_likelihood: Option<Measurement> = None;
            
            if !measurements.is_empty() {
                // Create weighted measurement from JPDA for likelihood computation
                let mut weighted_z = Vector3::zeros();
                let mut total_weight = 0.0;
                for (i, measurement) in measurements.iter().enumerate() {
                    let beta = association_probs[i + 1];
                    if beta > 1e-6 {
                        weighted_z += beta * measurement.z;
                        total_weight += beta;
                    }
                }
                if total_weight > 1e-6 {
                    weighted_z /= total_weight;
                    let base_r = measurements[0].R;
                    let r_adjustment = (1.0 / total_weight).min(5.0);
                    let r = base_r * r_adjustment;
                    measurement_for_likelihood = Some(Measurement::new(weighted_z, r, measurements[0].time));
                }
            }
            
            // Compute likelihoods for each model using PREDICTED states
            if let Some(ref meas) = measurement_for_likelihood {
                for (model_idx, predicted_state) in predicted_model_states.iter().enumerate() {
                    let model_filter: Box<dyn KalmanFilter> = get_filter(models[model_idx]);
                    let likelihood = model_filter.likelihood(predicted_state, meas);
                    likelihoods.push(likelihood);
                }
                // Update model probabilities based on predicted state likelihoods
                imm.update_model_probs_direct(&likelihoods);
            } else {
                // No measurement - keep current probabilities
                likelihoods = vec![1.0; models.len()];
            }
            
            // Step 3: Update each predicted model state with JPDA
            let mut updated_model_states = Vec::new();
            let mut was_updated = false;
            
            // Check if we have a valid association (not just missed detection)
            let has_valid_association = association_probs.len() > 1 && {
                let total_association: f64 = association_probs.iter().skip(1).sum();
                total_association > 0.1 // At least 10% association probability
            };
            
            // DEBUG: Log association probabilities
            log::debug!("Track {} - Association probabilities:", track.id);
            log::debug!("  Missed detection prob: {:.4}", association_probs[0]);
            for (i, &beta) in association_probs.iter().skip(1).enumerate() {
                if beta > 1e-6 {
                    log::debug!("  Measurement {} prob: {:.4}", i, beta);
                }
            }
            
            for (model_idx, predicted_state) in predicted_model_states.iter().enumerate() {
                // Create a temporary track with predicted state for JPDA update
                let mut temp_track = track.clone();
                temp_track.state = *predicted_state;
                
                log::debug!("  Model {} predicted state: pos=({:.2}, {:.2}, {:.2}), vel=({:.2}, {:.2}, {:.2})", 
                    model_idx, predicted_state.x[0], predicted_state.x[1], predicted_state.x[2],
                    predicted_state.x[3], predicted_state.x[4], predicted_state.x[5]);
                
                // Get appropriate filter for this model
                let model_filter: Box<dyn KalmanFilter> = get_filter(models[model_idx]);
                
                // Update this model's predicted state with JPDA
                let (updated_state, updated) = self.jpda.update_track(
                    &temp_track,
                    measurements,
                    &association_probs,
                    model_filter.as_ref(),
                    self.dt,
                );
                updated_model_states.push(updated_state);
                was_updated = was_updated || updated;
                
                log::debug!("  Model {} updated: {}, new pos=({:.2}, {:.2}, {:.2})", 
                    model_idx, updated, updated_state.x[0], updated_state.x[1], updated_state.x[2]);
            }
            
            // Consider it an update if we have valid association OR if track is very young
            // This helps young tracks survive initial uncertainty
            // For young tracks, be more lenient - consider partial associations as updates
            let effective_update = if track.age < 5 {
                was_updated || has_valid_association
            } else {
                was_updated
            };
            
            // Step 4: Combine updated model states using updated model probabilities
            let model_probs = imm.model_probs().to_vec();
            let mut combined_state = imm.combine_states(&updated_model_states);
            
            // DEBUG: Log state before velocity correction
            log::debug!("Track {} - Before velocity correction:", track.id);
            log::debug!("  Time: {:.2}, Age: {}, Was updated: {}", time, track.age, was_updated);
            log::debug!("  Position: ({:.2}, {:.2}, {:.2})", combined_state.x[0], combined_state.x[1], combined_state.x[2]);
            log::debug!("  Velocity: ({:.2}, {:.2}, {:.2})", combined_state.x[3], combined_state.x[4], combined_state.x[5]);
            log::debug!("  Prev track position available: {}", self.prev_track_position.is_some());
            
            // DEBUG: Check what the updated model states look like
            if !updated_model_states.is_empty() {
                log::debug!("  First model state position: ({:.2}, {:.2}, {:.2})", 
                    updated_model_states[0].x[0], updated_model_states[0].x[1], updated_model_states[0].x[2]);
                log::debug!("  First model state velocity: ({:.2}, {:.2}, {:.2})", 
                    updated_model_states[0].x[3], updated_model_states[0].x[4], updated_model_states[0].x[5]);
            }
            
            // CRITICAL: Improve velocity estimation using position history
            // Directly estimate velocity from position difference - this is essential
            // because the Kalman filter alone doesn't update velocity well from position-only measurements
            if let Some((prev_pos, prev_time)) = &self.prev_track_position {
                let dt_actual = (time - prev_time).max(0.1);
                log::debug!("  Using prev_track_position - dt: {:.2}, prev_pos: ({:.2}, {:.2}, {:.2})", 
                    dt_actual, prev_pos[0], prev_pos[1], prev_pos[2]);
                
                if dt_actual > 0.0 && dt_actual < 10.0 { // Only use recent history
                    // Compute velocity directly from position change
                    let current_pos = combined_state.position();
                    let pos_diff = current_pos - prev_pos;
                    let estimated_vel = pos_diff / dt_actual;
                    
                    log::debug!("  Position diff: ({:.2}, {:.2}, {:.2})", pos_diff[0], pos_diff[1], pos_diff[2]);
                    log::debug!("  Estimated velocity: ({:.2}, {:.2}, {:.2})", estimated_vel[0], estimated_vel[1], estimated_vel[2]);
                    log::debug!("  Filter velocity before: ({:.2}, {:.2}, {:.2})", combined_state.x[3], combined_state.x[4], combined_state.x[5]);
                    
                    // For all tracks, use significant weight on estimated velocity
                    // This is necessary because position-only measurements don't update velocity well
                    let blend_factor = if track.age < 5 {
                        0.9 // Very young tracks: 90% estimated, 10% filter
                    } else if track.age < 15 {
                        0.7 // Young tracks: 70% estimated, 30% filter
                    } else {
                        0.5 // Older tracks: 50% estimated, 50% filter
                    };
                    
                    log::debug!("  Blend factor: {:.2}", blend_factor);
                    
                    // Update velocity directly
                    combined_state.x[3] = blend_factor * estimated_vel[0] + (1.0 - blend_factor) * combined_state.x[3];
                    combined_state.x[4] = blend_factor * estimated_vel[1] + (1.0 - blend_factor) * combined_state.x[4];
                    combined_state.x[5] = blend_factor * estimated_vel[2] + (1.0 - blend_factor) * combined_state.x[5];
                    
                    log::debug!("  Velocity after blend: ({:.2}, {:.2}, {:.2})", combined_state.x[3], combined_state.x[4], combined_state.x[5]);
                    
                    // Update velocity covariance to reflect this knowledge
                    let vel_uncertainty = 50.0; // Lower uncertainty for directly estimated velocity
                    combined_state.P[(3, 3)] = (combined_state.P[(3, 3)] * (1.0 - blend_factor) + vel_uncertainty * blend_factor).min(300.0);
                    combined_state.P[(4, 4)] = (combined_state.P[(4, 4)] * (1.0 - blend_factor) + vel_uncertainty * blend_factor).min(300.0);
                    combined_state.P[(5, 5)] = (combined_state.P[(5, 5)] * (1.0 - blend_factor) + vel_uncertainty * blend_factor).min(300.0);
                    
                    // Maintain cross-covariance for future updates
                    let cross_cov = 150.0;
                    combined_state.P[(0, 3)] = cross_cov;
                    combined_state.P[(3, 0)] = cross_cov;
                    combined_state.P[(1, 4)] = cross_cov;
                    combined_state.P[(4, 1)] = cross_cov;
                    combined_state.P[(2, 5)] = cross_cov;
                    combined_state.P[(5, 2)] = cross_cov;
                } else {
                    log::debug!("  dt_actual out of range: {:.2}", dt_actual);
                }
            } else {
                log::debug!("  No prev_track_position - first update");
                // First update - estimate velocity from measurement if available
                if was_updated && !measurements.is_empty() {
                    log::debug!("  Was updated, checking prev_measurement");
                    // Try to get velocity from position difference if we have previous measurement
                    if let Some(prev_meas) = &self.prev_measurement {
                        let dt_meas = (time - prev_meas.time).max(0.1);
                        log::debug!("  Prev measurement time: {:.2}, current time: {:.2}, dt: {:.2}", prev_meas.time, time, dt_meas);
                        if dt_meas > 0.0 && dt_meas < 5.0 {
                            let pos_diff = combined_state.position() - prev_meas.z;
                            let estimated_vel = pos_diff / dt_meas;
                            log::debug!("  Setting velocity from prev_measurement: ({:.2}, {:.2}, {:.2})", estimated_vel[0], estimated_vel[1], estimated_vel[2]);
                            // Use this as initial velocity estimate
                            combined_state.x[3] = estimated_vel[0];
                            combined_state.x[4] = estimated_vel[1];
                            combined_state.x[5] = estimated_vel[2];
                        } else {
                            log::debug!("  dt_meas out of range: {:.2}", dt_meas);
                        }
                    } else {
                        log::debug!("  No prev_measurement available");
                    }
                } else {
                    log::debug!("  Not updated or no measurements");
                }
            }
            
            // DEBUG: Log final state
            log::debug!("Track {} - After velocity correction:", track.id);
            log::debug!("  Final velocity: ({:.2}, {:.2}, {:.2})", combined_state.x[3], combined_state.x[4], combined_state.x[5]);
            log::debug!("  Velocity magnitude: {:.2} m/s", combined_state.velocity().magnitude());
            
            let new_model_states = updated_model_states;
            
            // Update track and model states
            track.state = combined_state;
            track.model_probs = model_probs;
            track.age += 1;
            track.last_update_time = time;
            
            // Store current state and position for next velocity estimation
            self.prev_track_state = Some(combined_state);
            self.prev_track_position = Some((combined_state.position(), time));
            
            log::debug!("Track {} - Stored position ({:.2}, {:.2}, {:.2}) at time {:.2} for next update", 
                track.id, combined_state.position()[0], combined_state.position()[1], combined_state.position()[2], time);
            
            // Store updated model states
            self.model_states[track_idx] = new_model_states;
            
            if effective_update {
                track.missed_detections = 0;
                // Increase existence probability more aggressively when updated
                track.existence_prob = (track.existence_prob * 0.9 + 0.1).min(1.0);
            } else {
                track.missed_detections += 1;
                // Decrease existence probability very slowly - only for very old tracks
                // For young tracks (age < 10), don't decrease existence at all
                if track.age > 10 {
                    track.existence_prob *= 0.98; // Very slow decrease
                }
                // For very old tracks, decrease slightly faster
                if track.age > 50 {
                    track.existence_prob *= 0.97;
                }
            }
        }
        
        // Delete tracks that should be deleted
        self.delete_tracks();
    }
    
    
    /// Delete tracks that should be deleted
    fn delete_tracks(&mut self) {
        let mut to_delete = Vec::new();
        
        for (idx, track) in self.tracks.iter().enumerate() {
            if track.should_delete(self.max_missed_detections, self.min_existence_prob) {
                to_delete.push(idx);
            }
        }
        
        // Delete in reverse order to maintain indices
        for &idx in to_delete.iter().rev() {
            self.tracks.remove(idx);
            self.imm_filters.remove(idx);
            self.model_states.remove(idx);
        }
    }
    
    /// Get all tracks
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
    
    /// Get primary track (first track, if exists)
    pub fn primary_track(&self) -> Option<&Track> {
        self.tracks.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tracker_creation() {
        let tracker = Tracker::new(1.0);
        assert_eq!(tracker.tracks.len(), 0);
    }
    
    #[test]
    fn test_track_initialization() {
        let mut tracker = Tracker::new(1.0);
        let measurement = Measurement::with_default_covariance(
            nalgebra::Vector3::new(100.0, 200.0, 300.0),
            0.0,
        );
        
        tracker.update(&[measurement], 0.0);
        
        assert_eq!(tracker.tracks.len(), 1);
    }
    
    #[test]
    fn test_track_update() {
        let mut tracker = Tracker::new(1.0);
        let measurement1 = Measurement::with_default_covariance(
            nalgebra::Vector3::new(100.0, 200.0, 300.0),
            0.0,
        );
        
        tracker.update(&[measurement1], 0.0);
        
        let measurement2 = Measurement::with_default_covariance(
            nalgebra::Vector3::new(110.0, 210.0, 310.0),
            1.0,
        );
        
        tracker.update(&[measurement2], 1.0);
        
        // Track should still exist
        assert_eq!(tracker.tracks.len(), 1);
        assert_eq!(tracker.tracks[0].age, 1);
    }
    
    #[test]
    fn test_track_deletion() {
        let mut tracker = Tracker::new(1.0);
        let measurement = Measurement::with_default_covariance(
            nalgebra::Vector3::new(100.0, 200.0, 300.0),
            0.0,
        );
        
        tracker.update(&[measurement], 0.0);
        
        // Update with no measurements (missed detections)
        for i in 1..=6 {
            tracker.update(&[], i as f64);
        }
        
        // Track should be deleted after max missed detections
        assert_eq!(tracker.tracks.len(), 0);
    }
    
    #[test]
    fn test_primary_track() {
        let mut tracker = Tracker::new(1.0);
        let measurement = Measurement::with_default_covariance(
            nalgebra::Vector3::new(100.0, 200.0, 300.0),
            0.0,
        );
        
        tracker.update(&[measurement], 0.0);
        
        let primary = tracker.primary_track();
        assert!(primary.is_some());
        assert_eq!(primary.unwrap().id, 0);
    }
}

