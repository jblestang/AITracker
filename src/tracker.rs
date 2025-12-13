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
use rayon::prelude::*;
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
    /// Expected number of targets (from simulation)
    expected_num_targets: usize,
}

impl Tracker {
    /// Create a new tracker
    /// 
    /// # Arguments
    /// * `dt` - Time step
    /// * `expected_num_targets` - Expected number of targets (from simulation)
    pub fn new(dt: f64, expected_num_targets: usize) -> Self {
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
            expected_num_targets,
        }
    }
    
    /// Set expected number of targets (can be updated if simulation changes)
    pub fn set_expected_num_targets(&mut self, num: usize) {
        self.expected_num_targets = num;
    }
    
    /// Compute track score for a measurement (log-likelihood ratio)
    /// 
    /// Track score compares the likelihood of the measurement being from a real target
    /// vs. being clutter. Higher scores indicate more likely real targets.
    /// 
    /// Based on state-of-the-art track initialization techniques:
    /// - Uses log-likelihood ratio: log(P(target|measurement) / P(clutter|measurement))
    /// - Accounts for clutter density and detection probability
    /// - Validates velocity consistency
    /// 
    /// # Arguments
    /// * `measurement` - Measurement to score
    /// * `time` - Current time
    /// 
    /// # Returns
    /// Track score (higher = more likely to be a real target)
    fn compute_track_score(&self, measurement: &Measurement, time: f64) -> f64 {
        // State-of-the-art track scoring uses log-likelihood ratio
        // Score = log(P(target|z) / P(clutter|z))
        // 
        // Using simplified approach:
        // - If we have previous measurement, compute velocity and validate it
        // - Higher score for measurements with reasonable velocity estimates
        // - Lower score for isolated measurements (more likely clutter)
        
        let mut score = 0.0;
        
        // Base score: detection probability vs clutter probability
        // P(target) = pd (detection probability)
        // P(clutter) = lambda_fa * V (expected false alarms in volume)
        // For a single measurement, we use a simplified model
        
        // If we have a previous measurement, estimate velocity and validate
        if let Some(prev_meas) = &self.prev_measurement {
            let dt = (measurement.time - prev_meas.time).max(0.1);
            if dt > 0.0 && dt < 10.0 {
                let estimated_velocity = (measurement.z - prev_meas.z) / dt;
                let speed = estimated_velocity.magnitude();
                
                // Velocity gate: reasonable aircraft speeds (0-300 m/s ≈ 0-1080 km/h)
                // Higher score for speeds in typical aircraft range (50-250 m/s)
                if speed > 0.0 && speed < 300.0 {
                    // Score based on how reasonable the speed is
                    if speed >= 50.0 && speed <= 250.0 {
                        score += 2.0; // Good speed range
                    } else if speed >= 20.0 && speed < 50.0 {
                        score += 1.0; // Slow but possible
                    } else if speed > 250.0 && speed < 300.0 {
                        score += 1.0; // Fast but possible
                    } else {
                        score -= 1.0; // Unlikely speed
                    }
                } else {
                    // Unreasonable speed - likely clutter
                    return -10.0; // Very low score
                }
            }
        } else {
            // No previous measurement - isolated measurement
            // Lower score (more likely to be clutter)
            score -= 1.0;
        }
        
        // Account for clutter density
        // Higher clutter density -> lower score threshold needed
        // Lower clutter density -> can be more selective
        let clutter_penalty = self.jpda.lambda_fa * 1e6; // Approximate volume penalty
        score -= clutter_penalty.ln().max(0.0) * 0.1;
        
        // Detection probability bonus
        score += self.jpda.pd.ln() * 0.5;
        
        score
    }
    
    /// Initialize a new track
    /// 
    /// State-of-the-art track initialization with quality filtering:
    /// - Track score threshold to filter out likely clutter
    /// - Velocity validation
    /// - M/N confirmation logic (3 valid associations required)
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
        
        // Improved velocity estimation: use least-squares fit if we have measurement history
        log::info!("[TRACK INIT] Initializing track {} at time {:.2}", self.next_track_id, time);
        log::info!("  Measurement position: ({:.2}, {:.2}, {:.2})", measurement.z[0], measurement.z[1], measurement.z[2]);
        
        // Try to estimate velocity using least-squares from recent measurements
        let mut estimated_velocity = Vector3::zeros();
        let mut has_velocity_estimate = false;
        
        if let Some(prev_meas) = &self.prev_measurement {
            let dt_meas = (measurement.time - prev_meas.time).max(0.1);
            if dt_meas > 0.0 && dt_meas < 10.0 {
                // Simple two-point estimate
                estimated_velocity = (measurement.z - prev_meas.z) / dt_meas;
                has_velocity_estimate = true;
                log::info!("  Two-point velocity estimate: ({:.2}, {:.2}, {:.2}) m/s", 
                    estimated_velocity[0], estimated_velocity[1], estimated_velocity[2]);
            }
        }
        
        // If we have measurement history, use least-squares fit for better estimate
        // This would require storing measurement history, which we'll add to Track
        // For now, use the two-point estimate if available
        
        if has_velocity_estimate {
            x[3] = estimated_velocity[0];
            x[4] = estimated_velocity[1];
            x[5] = estimated_velocity[2];
        } else {
            log::info!("  No velocity estimate available - initializing to zero");
        }
        
        // Initial covariance: position uncertainty from measurement, velocity uncertainty larger
        let mut P = nalgebra::Matrix6::identity() * 1000.0;
        P[(0, 0)] = measurement.R[(0, 0)];
        P[(1, 1)] = measurement.R[(1, 1)];
        P[(2, 2)] = measurement.R[(2, 2)];
        
        // Velocity uncertainty: adaptive based on estimation quality
        let vel_uncertainty = if has_velocity_estimate {
            // Estimated velocity - moderate uncertainty, scaled by time step
            let dt = (measurement.time - self.prev_measurement.unwrap().time).max(0.1);
            // Uncertainty increases with time step (less reliable for larger dt)
            (200.0 * (1.0 + dt / 2.0)).min(500.0)
        } else {
            1000.0 // No velocity estimate - large uncertainty
        };
        P[(3, 3)] = vel_uncertainty;
        P[(4, 4)] = vel_uncertainty;
        P[(5, 5)] = vel_uncertainty;
        
        // Cross-covariance terms (position-velocity correlation)
        // Increased cross-covariance to help velocity convergence
        // Scale with velocity uncertainty for better correlation
        let cross_cov = (vel_uncertainty * 0.5).min(300.0).max(200.0);
        P[(0, 3)] = cross_cov;
        P[(3, 0)] = cross_cov;
        P[(1, 4)] = cross_cov;
        P[(4, 1)] = cross_cov;
        P[(2, 5)] = cross_cov;
        P[(5, 2)] = cross_cov;
        
        let initial_state = State::new(x, P);
        
        // Create IMM filter with adaptive transition probabilities
        let imm = IMM::default();
        let num_models = imm.model_probs().len();
        
        // Create track (starts as tentative, will be confirmed with M/N logic)
        let mut track = Track::new(self.next_track_id, initial_state, num_models, time);
        track.add_measurement(measurement.z, time);
        track.increment_scan();
        self.next_track_id += 1;
        
        // Initialize model-conditioned states (one per model)
        let model_states = vec![initial_state; num_models];
        
        self.tracks.push(track);
        self.imm_filters.push(imm);
        self.model_states.push(model_states);
        
        log::info!("[TRACK INIT] Track {} initialized with velocity: ({:.2}, {:.2}, {:.2}) m/s", 
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
        
        // If we have no tracks, initialize tracks from measurements
        // Otherwise, update existing tracks and then initialize new ones from unassociated measurements
        if self.tracks.is_empty() {
            // Initialize tracks from measurements (up to a limit)
            // Limit to expected number of targets
            let max_initial_tracks = self.expected_num_targets;
            log::info!("[TRACK INIT] No existing tracks. Initializing up to {} tracks from {} measurements (expected {} targets)", 
                max_initial_tracks, measurements.len(), self.expected_num_targets);
            
            // Use distance-based clustering to select distinct initial measurements
            // This helps when there are many measurements (clutter + true)
            let mut selected_measurements = Vec::new();
            let min_separation = 500.0; // Minimum 500m separation between initial tracks
            
            for measurement in measurements.iter() {
                if selected_measurements.len() >= max_initial_tracks {
                    break;
                }
                
                // Check if this measurement is far enough from already selected ones
                let is_far_enough = selected_measurements.iter().all(|(m, _idx): &(Measurement, usize)| {
                    (measurement.z - m.z).magnitude() >= min_separation
                });
                
                if is_far_enough {
                    selected_measurements.push((*measurement, selected_measurements.len()));
                }
            }
            
            // If we don't have enough distinct measurements, take the first N anyway
            if selected_measurements.len() < max_initial_tracks {
                for (i, measurement) in measurements.iter().enumerate() {
                    if selected_measurements.len() >= max_initial_tracks {
                        break;
                    }
                    // Check if not already selected
                    if !selected_measurements.iter().any(|(m, _idx): &(Measurement, usize)| (m.z - measurement.z).magnitude() < 10.0) {
                        selected_measurements.push((*measurement, selected_measurements.len()));
                    }
                }
            }
            
            for (i, (measurement, _)) in selected_measurements.iter().enumerate() {
                log::info!("[TRACK INIT] Initializing track {} from measurement at ({:.1}, {:.1}, {:.1})", 
                    i, measurement.z[0], measurement.z[1], measurement.z[2]);
                self.initialize_track(measurement, time);
            }
            log::info!("[TRACK INIT] Initialized {} tracks total (expected {})", 
                self.tracks.len(), self.expected_num_targets);
            // Reset previous track state when initializing new tracks
            self.prev_track_state = None;
            self.prev_track_position = None;
            // DON'T return early - continue to process remaining measurements for additional tracks
            // The initial creation might not have created all expected tracks if measurements were too close
        }
        
        log::info!("[TRACK UPDATE] Updating {} existing tracks with {} measurements", 
            self.tracks.len(), measurements.len());
        
        // Log all track positions and measurement positions for debugging
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let track_pos = track.state.position();
            log::info!("[TRACK {}] Position: ({:.1}, {:.1}, {:.1}), Age: {}, Missed: {}", 
                track.id, track_pos[0], track_pos[1], track_pos[2], track.age, track.missed_detections);
        }
        for (meas_idx, measurement) in measurements.iter().enumerate() {
            log::info!("[MEAS {}] Position: ({:.1}, {:.1}, {:.1})", 
                meas_idx, measurement.z[0], measurement.z[1], measurement.z[2]);
        }
        
        // Check for conflicts when multiple tracks associate with the same measurement
        // Use joint probabilities to resolve conflicts properly
        // FIX: Remove measurement count requirement - conflicts can occur with 1 measurement
        // FIX: Pre-compute associations once and reuse them
        let mut resolved_associations: Option<Vec<Vec<f64>>> = None;
        let mut initial_associations: Vec<Vec<f64>> = Vec::new();
        
        if !measurements.is_empty() {
            // Pre-compute association probabilities for all tracks (always, for reuse)
            let filter_for_gating: Box<dyn KalmanFilter> = get_filter(MotionModel::ConstantVelocity);
            for track in &self.tracks {
                let assoc_probs = self.jpda.compute_association_probs(
                    track,
                    measurements,
                    filter_for_gating.as_ref(),
                    self.dt,
                );
                initial_associations.push(assoc_probs);
            }
        }
        
        if self.tracks.len() >= 2 && !measurements.is_empty() {
            
            // Check for conflicts: multiple tracks associating with same measurement
            // Adaptive threshold: lower for more tracks (more competition)
            let conflict_threshold = if self.tracks.len() >= 3 {
                0.15 // Lower threshold for 3+ tracks
            } else {
                0.2 // Standard threshold for 2 tracks
            };
            
            let mut has_conflict = false;
            let mut conflict_measurements: Vec<(usize, Vec<(usize, f64)>)> = Vec::new();
            
            for meas_idx in 0..measurements.len() {
                let mut tracks_associating = Vec::new();
                for (track_idx, assoc_probs) in initial_associations.iter().enumerate() {
                    if assoc_probs[meas_idx + 1] > conflict_threshold {
                        tracks_associating.push((track_idx, assoc_probs[meas_idx + 1]));
                    }
                }
                if tracks_associating.len() > 1 {
                    has_conflict = true;
                    let tracks_info: Vec<String> = tracks_associating.iter()
                        .map(|(i, p)| format!("Track{}:{:.3}", i, p))
                        .collect();
                    log::warn!("[CONFLICT] MEAS {} has {} tracks associating: {:?}", 
                        meas_idx, tracks_associating.len(), tracks_info);
                    conflict_measurements.push((meas_idx, tracks_associating));
                }
            }
            
            // Resolve conflicts using improved assignment for 3+ tracks
            if has_conflict {
                let mut resolved = initial_associations.clone();
                let filter_for_gating: Box<dyn KalmanFilter> = get_filter(MotionModel::ConstantVelocity);
                
                // For 3+ tracks, try to balance assignments so each track gets at least one good measurement
                // Track which measurements have been assigned and which tracks need assignments
                let mut track_has_good_assignment = vec![false; self.tracks.len()];
                let mut measurement_assigned = vec![false; measurements.len()];
                
                // First pass: assign measurements to tracks that need them most
                // Sort conflicts by number of competing tracks (handle larger conflicts first)
                let mut sorted_conflicts: Vec<(usize, Vec<(usize, f64)>)> = conflict_measurements.iter()
                    .map(|(meas_idx, tracks_associating)| (*meas_idx, tracks_associating.clone()))
                    .collect();
                sorted_conflicts.sort_by(|(_, a), (_, b)| b.len().cmp(&a.len()));
                
                for (meas_idx, tracks_associating) in sorted_conflicts {
                    if tracks_associating.len() < 2 {
                        continue;
                    }
                    
                    // Compute likelihoods for all competing tracks
                    let mut track_scores = Vec::new();
                    for (track_idx, assoc_prob) in &tracks_associating {
                        let pred = filter_for_gating.predict(&self.tracks[*track_idx].state, self.dt);
                        let likelihood = filter_for_gating.likelihood(&pred, &measurements[meas_idx]);
                        
                        // Consider track consistency (confirmed tracks are preferred)
                        let track_consistent = self.tracks[*track_idx].is_confirmed && 
                                               self.tracks[*track_idx].missed_detections < 3;
                        let consistency_bonus = if track_consistent { 1.2 } else { 1.0 };
                        
                        // FIX: Prioritize tracks without assignments, penalize tracks with assignments
                        let assignment_bonus = if self.tracks.len() >= 3 && !track_has_good_assignment[*track_idx] {
                            1.3 // Bonus for tracks without assignments (30% boost)
                        } else {
                            1.0
                        };
                        let assignment_penalty = if self.tracks.len() >= 3 && track_has_good_assignment[*track_idx] {
                            0.7 // Reduce score if track already has a good assignment
                        } else {
                            1.0
                        };
                        
                        let adjusted_score = likelihood * consistency_bonus * assignment_bonus * assignment_penalty;
                        track_scores.push((*track_idx, adjusted_score, likelihood, *assoc_prob));
                    }
                    
                    // Find track with highest score
                    let best_track = track_scores.iter()
                        .max_by(|(_, a, _, _), (_, b, _, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _, _, _)| *idx);
                    
                    if let Some(best_idx) = best_track {
                        let best_score = track_scores.iter()
                            .find(|(idx, _, _, _)| *idx == best_idx)
                            .map(|(_, _, l, _)| *l)
                            .unwrap_or(0.0);
                        
                        // Assign measurement to best track
                        let best_assoc = initial_associations[best_idx][meas_idx + 1];
                        resolved[best_idx][meas_idx + 1] = (best_assoc * 1.2).min(0.95);
                        track_has_good_assignment[best_idx] = true;
                        measurement_assigned[meas_idx] = true;
                        
                        // Redistribute probabilities: reduce other tracks' associations
                        // Use adaptive reduction based on number of tracks
                        let reduction_factor = if self.tracks.len() >= 3 {
                            0.2 // More aggressive reduction for 3+ tracks (20% of original)
                        } else {
                            0.3 // Standard reduction for 2 tracks (30% of original)
                        };
                        
                        for (track_idx, _, _, _) in &track_scores {
                            if *track_idx != best_idx {
                                resolved[*track_idx][meas_idx + 1] *= reduction_factor;
                            }
                        }
                        
                        // Normalize all affected tracks
                        for (track_idx, _, _, _) in &track_scores {
                            let sum: f64 = resolved[*track_idx].iter().sum();
                            if sum > 1e-10 {
                                for prob in &mut resolved[*track_idx] {
                                    *prob /= sum;
                                }
                            } else {
                                resolved[*track_idx][0] = 1.0;
                                for i in 1..resolved[*track_idx].len() {
                                    resolved[*track_idx][i] = 0.0;
                                }
                            }
                        }
                        
                        log::info!("[CONFLICT RESOLVED] Track {} gets MEAS {} (score: {:.3e}, likelihood: {:.3e})", 
                            best_idx, meas_idx, best_score, best_score);
                    }
                }
                
                resolved_associations = Some(resolved);
            } else {
                // No conflict detected, use original associations
                resolved_associations = Some(initial_associations.clone());
            }
        } else {
            // Single track or no measurements: use pre-computed associations
            resolved_associations = Some(initial_associations.clone());
        }
        
        // Update existing tracks
        // Note: We parallelize association probability computation where possible,
        // but IMM updates must remain sequential due to trait object constraints
        for (track_idx, track) in self.tracks.iter_mut().enumerate() {
            // Get IMM filter and model states for this track
            let imm = &mut self.imm_filters[track_idx];
            let current_model_states = self.model_states[track_idx].clone();
            
            // Get filters for each model
            let models = MotionModel::all();
            
            // FIX: Reuse pre-computed association probabilities from conflict detection
            // This avoids double computation and ensures consistency
            let filter_for_gating: Box<dyn KalmanFilter> = get_filter(MotionModel::ConstantVelocity);
            let mut association_probs = if let Some(ref resolved) = resolved_associations {
                // Use resolved associations if available (for conflict resolution)
                resolved[track_idx].clone()
            } else if track_idx < initial_associations.len() {
                // Reuse pre-computed associations from conflict detection phase
                initial_associations[track_idx].clone()
            } else {
                // Fallback: compute associations independently (shouldn't happen normally)
                self.jpda.compute_association_probs(
                    track,
                    measurements,
                    filter_for_gating.as_ref(),
                    self.dt,
                )
            };
            
            // Ensure association probabilities are valid (sum to 1, all non-negative)
            let sum: f64 = association_probs.iter().sum();
            if sum > 1e-10 && (sum - 1.0).abs() > 0.01 {
                // Renormalize if sum is significantly different from 1.0
                for prob in &mut association_probs {
                    *prob /= sum;
                }
                log::debug!("[TRACK {}] Renormalized association probabilities (sum was {:.4})", track.id, sum);
            }
            
            // Log association probabilities for this track
            log::info!("[TRACK {}] Association probabilities: missed={:.3}", 
                track.id, association_probs[0]);
            let track_pos = track.state.position();
            for (meas_idx, &beta) in association_probs.iter().skip(1).enumerate() {
                let meas_pos = measurements[meas_idx].z;
                let distance = (meas_pos - track_pos).magnitude();
                if beta > 0.01 {
                    log::info!("[TRACK {}]   -> MEAS {}: prob={:.3}, distance={:.1}m", 
                        track.id, meas_idx, beta, distance);
                } else {
                    log::debug!("[TRACK {}]   -> MEAS {}: prob={:.3}, distance={:.1}m", 
                        track.id, meas_idx, beta, distance);
                }
            }
            
            // Find the measurement with highest association probability
            let max_assoc_idx = association_probs.iter()
                .enumerate()
                .skip(1)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx - 1);
            
            if let Some(best_meas_idx) = max_assoc_idx {
                let best_beta = association_probs[best_meas_idx + 1];
                if best_beta > 0.3 {
                    let best_meas_pos = measurements[best_meas_idx].z;
                    let distance_to_best = (best_meas_pos - track_pos).magnitude();
                    log::info!("[TRACK {}] PRIMARY ASSOCIATION: MEAS {} with prob={:.3}, distance={:.1}m", 
                        track.id, best_meas_idx, best_beta, distance_to_best);
                }
            }
            
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
            // No measurement - keep current probabilities (don't update model probs)
            // likelihoods remains empty, which is fine
        }
        
        // Step 3: Update each predicted model state with JPDA
        let mut updated_model_states = Vec::new();
        let mut was_updated = false;
        
        // Check if we have a valid association (not just missed detection)
        let has_valid_association = association_probs.len() > 1 && {
            let total_association: f64 = association_probs.iter().skip(1).sum();
            total_association > 0.05 // At least 5% association probability (lowered from 0.1)
        };
        
        for (model_idx, predicted_state) in predicted_model_states.iter().enumerate() {
            // Create a temporary track with predicted state for JPDA update
            let mut temp_track = track.clone();
            temp_track.state = *predicted_state;
            
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
        }
        
        // Consider it an update if we have valid association OR if JPDA updated
        // Be more lenient: count any association > 5% as an update, not just when JPDA returns updated=true
        // This prevents missed detections from accumulating when tracks have weak but valid associations
        let effective_update = was_updated || has_valid_association;
        
        // Step 4: Combine updated model states using updated model probabilities
        let model_probs = imm.model_probs().to_vec();
        let mut combined_state = imm.combine_states(&updated_model_states);
        
        // Log the update result
        let old_pos = track.state.position();
        let new_pos = combined_state.position();
        let pos_change = (new_pos - old_pos).magnitude();
        log::info!("[TRACK {}] Update: old_pos=({:.1}, {:.1}, {:.1}), new_pos=({:.1}, {:.1}, {:.1}), change={:.1}m", 
            track.id, old_pos[0], old_pos[1], old_pos[2], 
            new_pos[0], new_pos[1], new_pos[2], pos_change);
        
        // FIX: Use track-specific previous position for velocity estimation
        // This fixes the bug where all tracks used the last track's position
        if let Some((prev_pos, prev_time)) = &track.prev_position {
            let dt_actual = (time - prev_time).max(0.1);
            
            if dt_actual > 0.0 && dt_actual < 10.0 {
                // Compute velocity directly from position change
                let current_pos = combined_state.position();
                let pos_diff = current_pos - prev_pos;
                let estimated_vel = pos_diff / dt_actual;
                
                // Blend estimated velocity with filter velocity
                let blend_factor = if track.age < 5 {
                    0.9
                } else if track.age < 15 {
                    0.7
                } else {
                    0.5
                };
                
                // Update velocity
                combined_state.x[3] = blend_factor * estimated_vel[0] + (1.0 - blend_factor) * combined_state.x[3];
                combined_state.x[4] = blend_factor * estimated_vel[1] + (1.0 - blend_factor) * combined_state.x[4];
                combined_state.x[5] = blend_factor * estimated_vel[2] + (1.0 - blend_factor) * combined_state.x[5];
                
                // Update velocity covariance
                let vel_uncertainty = 50.0;
                combined_state.P[(3, 3)] = (combined_state.P[(3, 3)] * (1.0 - blend_factor) + vel_uncertainty * blend_factor).min(300.0);
                combined_state.P[(4, 4)] = (combined_state.P[(4, 4)] * (1.0 - blend_factor) + vel_uncertainty * blend_factor).min(300.0);
                combined_state.P[(5, 5)] = (combined_state.P[(5, 5)] * (1.0 - blend_factor) + vel_uncertainty * blend_factor).min(300.0);
                
                // Maintain cross-covariance
                let cross_cov = 150.0;
                combined_state.P[(0, 3)] = cross_cov;
                combined_state.P[(3, 0)] = cross_cov;
                combined_state.P[(1, 4)] = cross_cov;
                combined_state.P[(4, 1)] = cross_cov;
                combined_state.P[(2, 5)] = cross_cov;
                combined_state.P[(5, 2)] = cross_cov;
            }
        } else if effective_update && !measurements.is_empty() {
            // First update - estimate velocity from measurement if available
            if let Some(prev_meas) = &self.prev_measurement {
                let dt_meas = (time - prev_meas.time).max(0.1);
                if dt_meas > 0.0 && dt_meas < 5.0 {
                    let pos_diff = combined_state.position() - prev_meas.z;
                    let estimated_vel = pos_diff / dt_meas;
                    combined_state.x[3] = estimated_vel[0];
                    combined_state.x[4] = estimated_vel[1];
                    combined_state.x[5] = estimated_vel[2];
                }
            }
        }
        
            // Update track
            track.state = combined_state;
            track.model_probs = model_probs;
            track.age += 1;
            track.last_update_time = time;
            
            // FIX: Store track-specific previous position (not shared across tracks)
            // Store current state and position for next velocity estimation
            self.prev_track_state = Some(combined_state);
            track.prev_position = Some((combined_state.position(), time));
            
            // Store updated model states
            self.model_states[track_idx] = updated_model_states;
            
            if effective_update {
                track.missed_detections = 0;
                // Increase existence probability more aggressively when updated
                track.existence_prob = (track.existence_prob * 0.9 + 0.1).min(1.0);
                
                // Add measurement to history for velocity estimation and track confirmation
                // Count this as a valid association for confirmation logic (3 valid associations needed)
                // Since effective_update is true, this means the track was actually updated with a measurement
                if let Some(best_meas_idx) = association_probs.iter()
                    .enumerate()
                    .skip(1)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx - 1)
                {
                    if best_meas_idx < measurements.len() {
                        // Count as valid association - effective_update means we have a valid association
                        // Use a very low threshold (0.05) to count most associations, but still filter out noise
                        if association_probs[best_meas_idx + 1] > 0.05 {
                            track.add_measurement(measurements[best_meas_idx].z, time);
                            log::debug!("[TRACK {}] Valid association #{} (prob={:.3})", 
                                track.id, track.num_detections, association_probs[best_meas_idx + 1]);
                        }
                    }
                }
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
            
            // Increment scan counter for M/N logic
            track.increment_scan();
            
            // NEW LOGIC: Track becomes active/confirmed after 3 valid associations
            // A valid association means the track was actually updated (effective_update = true)
            // This is tracked via num_detections which is incremented in add_measurement
            if !track.is_confirmed && track.num_detections >= 3 {
                track.is_confirmed = true;
                log::info!("[TRACK CONFIRM] Track {} confirmed after {} valid associations", 
                    track.id, track.num_detections);
            }
            
            // Improve velocity estimation using measurement history (least-squares)
            if track.measurement_history.len() >= 2 && !track.measurement_history.is_empty() {
                // Use least-squares to estimate velocity from position history
                let positions: Vec<Vector3<f64>> = track.measurement_history.iter()
                    .map(|(pos, _)| *pos).collect();
                let times: Vec<f64> = track.measurement_history.iter()
                    .map(|(_, t)| *t).collect();
                
                // Simple least-squares: fit line to positions vs time
                let n = positions.len() as f64;
                let sum_t: f64 = times.iter().sum();
                
                // For each dimension, compute velocity = sum((t - t_mean) * (pos - pos_mean)) / sum((t - t_mean)^2)
                let t_mean = sum_t / n;
                let mut vel_estimate = Vector3::zeros();
                
                for dim in 0..3 {
                    let sum_pos: f64 = positions.iter().map(|p| p[dim]).sum();
                    let pos_mean = sum_pos / n;
                    
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for i in 0..positions.len() {
                        let t_diff = times[i] - t_mean;
                        let pos_diff = positions[i][dim] - pos_mean;
                        numerator += t_diff * pos_diff;
                        denominator += t_diff * t_diff;
                    }
                    
                    if denominator > 1e-6 {
                        vel_estimate[dim] = numerator / denominator;
                    }
                }
                
                // Blend velocity estimate with filter estimate (more weight on history for young tracks)
                let blend_factor = if track.age < 5 {
                    0.7 // 70% weight on history-based estimate for young tracks
                } else {
                    0.3 // 30% weight for mature tracks
                };
                
                let filter_vel = track.state.velocity();
                let blended_vel = blend_factor * vel_estimate + (1.0 - blend_factor) * filter_vel;
                
                // Update velocity in state
                track.state.x[3] = blended_vel[0];
                track.state.x[4] = blended_vel[1];
                track.state.x[5] = blended_vel[2];
                
                log::debug!("[VEL EST] Track {}: history={:.2?}, filter={:.2?}, blended={:.2?}", 
                    track.id, vel_estimate, filter_vel, blended_vel);
            }
            
            // Covariance inflation during maneuvers (when model probabilities are changing)
            let model_prob_variance: f64 = track.model_probs.iter()
                .map(|p| (p - 1.0 / track.model_probs.len() as f64).powi(2))
                .sum();
            let model_uncertainty = model_prob_variance.sqrt();
            
            // Inflate covariance if model probabilities are uncertain (indicating maneuver)
            if model_uncertainty > 0.15 { // High uncertainty = maneuver
                let inflation_factor = 1.0 + model_uncertainty * 0.5; // Up to 50% inflation
                track.state.P = track.state.P * inflation_factor;
                log::debug!("[COV INFLATION] Track {}: uncertainty={:.3}, inflation={:.2}", 
                    track.id, model_uncertainty, inflation_factor);
            }
        }
        
        // Check if tracks are too close together (might be tracking same target)
        // Also check if tracks are associating with the same measurements
        if self.tracks.len() >= 2 {
            for i in 0..self.tracks.len() {
                for j in (i+1)..self.tracks.len() {
                    let pos_i = self.tracks[i].state.position();
                    let pos_j = self.tracks[j].state.position();
                    let distance = (pos_i - pos_j).magnitude();
                    if distance < 500.0 {
                        log::warn!("[TRACK PROXIMITY] Track {} and Track {} are very close: {:.1}m apart - might be tracking same target!", 
                            self.tracks[i].id, self.tracks[j].id, distance);
                    }
                }
            }
            
            // Check if multiple tracks are associating with the same measurement
            if !measurements.is_empty() {
                for (meas_idx, measurement) in measurements.iter().enumerate() {
                    let mut tracks_associating = Vec::new();
                    for track in &self.tracks {
                        // Recompute association to see which tracks want this measurement
                        let filter_for_gating: Box<dyn KalmanFilter> = get_filter(MotionModel::ConstantVelocity);
                        let assoc_probs = self.jpda.compute_association_probs(
                            track,
                            &[measurement.clone()],
                            filter_for_gating.as_ref(),
                            self.dt,
                        );
                        if assoc_probs.len() > 1 && assoc_probs[1] > 0.3 {
                            tracks_associating.push((track.id, assoc_probs[1]));
                        }
                    }
                    if tracks_associating.len() > 1 {
                        log::warn!("[MEASUREMENT CONFLICT] MEAS {} is being associated with {} tracks: {:?}", 
                            meas_idx, tracks_associating.len(), tracks_associating);
                    }
                }
            }
        }
        
        // Delete tracks that should be deleted
        self.delete_tracks();
        
        // Step 4: Initialize new tracks from unassociated measurements
        // For multi-target tracking, we need to create tracks for measurements
        // that are not well-associated with existing tracks
        self.initialize_new_tracks_from_unassociated(measurements, time);
    }
    
    /// Initialize new tracks from unassociated measurements
    /// 
    /// A measurement is considered unassociated if it has low association probability
    /// with all existing tracks (i.e., it's likely a new target or clutter).
    /// 
    /// # Arguments
    /// * `measurements` - All measurements (true + clutter)
    /// * `time` - Current time
    fn initialize_new_tracks_from_unassociated(&mut self, measurements: &[Measurement], time: f64) {
        if measurements.is_empty() {
            return;
        }
        
        // For each measurement, check if it's associated with any existing track
        // A measurement is unassociated if its max association probability with any track is low
        // Parallelize this check for better performance
        let tracks_ref = &self.tracks;
        let jpda_ref = &self.jpda;
        let dt = self.dt;
        
        // Consider all tracks (both confirmed and pending) for association
        // Pending tracks can also associate with measurements to become confirmed
        let all_tracks: Vec<&Track> = tracks_ref.iter().collect();
        
        let unassociated_measurements: Vec<(usize, Measurement)> = measurements
            .par_iter()
            .enumerate()
            .filter_map(|(meas_idx, measurement)| {
                let max_association: f64 = all_tracks
                    .par_iter()
                    .map(|track| {
                        let filter_for_gating: Box<dyn KalmanFilter> = get_filter(MotionModel::ConstantVelocity);
                        let association_probs = jpda_ref.compute_association_probs(
                            track,
                            &[measurement.clone()], // Single measurement
                            filter_for_gating.as_ref(),
                            dt,
                        );
                        
                        // Get association probability for this measurement (skip missed detection prob)
                        if association_probs.len() > 1 {
                            association_probs[1] // First measurement prob
                        } else {
                            0.0
                        }
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                
                // Adaptive threshold based on number of tracks and expected targets
                // With more tracks, individual association probabilities may be lower
                // Lower threshold for 3+ tracks to allow proper association
                let association_threshold = if self.expected_num_targets >= 3 {
                    0.3 // Lower threshold for 3+ targets
                } else if self.expected_num_targets == 2 {
                    0.4 // Medium threshold for 2 targets
                } else {
                    0.5 // Higher threshold for 1 target (very conservative)
                };
                
                // If max association is low, consider it unassociated
                if max_association < association_threshold {
                    log::debug!("[UNASSOC] Measurement {} at ({:.1}, {:.1}, {:.1}) has max_association={:.3} < 0.5", 
                        meas_idx, measurement.z[0], measurement.z[1], measurement.z[2], max_association);
                    Some((meas_idx, measurement.clone()))
                } else {
                    log::debug!("[ASSOC] Measurement {} at ({:.1}, {:.1}, {:.1}) is associated (max_association={:.3})", 
                        meas_idx, measurement.z[0], measurement.z[1], measurement.z[2], max_association);
                    None
                }
            })
            .collect();
        
        // NEW LOGIC: Every unassociated measurement creates a new pending track
        // Tracks become active/confirmed after 3 valid associations
        log::info!("[TRACK INIT] Found {} unassociated measurements, {} existing tracks, creating pending tracks for all unassociated measurements", 
            unassociated_measurements.len(), self.tracks.len());
        
        // Check distance to existing tracks to avoid duplicates
        // Only create track if measurement is far enough from existing tracks
        let min_separation = 200.0; // Minimum 200m separation to avoid duplicate tracks
        let tracks_ref = &self.tracks;
        
        // State-of-the-art track initialization: filter by quality
        // 1. Distance check (avoid duplicates)
        // 2. Track score threshold (filter likely clutter)
        // 3. Velocity validation (reasonable speeds)
        
        // More lenient score threshold for initial tracks
        // Since we're already filtering by distance and unassociation, we can be less strict
        let min_track_score = 0.0; // Minimum score to initialize track (more lenient)
        let prev_meas_ref = &self.prev_measurement;
        let jpda_pd = self.jpda.pd;
        let jpda_lambda_fa = self.jpda.lambda_fa;
        
        let measurements_to_initialize: Vec<(Measurement, f64)> = unassociated_measurements
            .iter()
            .filter_map(|(meas_idx, measurement)| {
                // Check distance to all existing tracks (both confirmed and pending)
                let is_far_enough = tracks_ref.iter().all(|track| {
                    let track_pos = track.state.position();
                    (measurement.z - track_pos).magnitude() >= min_separation
                });
                
                if !is_far_enough {
                    log::debug!("[TRACK INIT] Skipping measurement {} - too close to existing track", meas_idx);
                    return None;
                }
                
                // Compute track score (log-likelihood ratio)
                // CRITICAL FIX: Don't use shared prev_measurement for velocity validation
                // because it's from a different target. Instead, use a more lenient approach
                // for initial track creation.
                let mut score = 0.0;
                
                // For initial track creation, we can't reliably compute velocity from
                // a shared previous measurement (it might be from a different aircraft).
                // Instead, use a more lenient scoring approach:
                // - If we have existing tracks, this measurement is far enough away (already checked)
                // - Give it a base score that allows initialization
                // - Velocity validation will happen after the track is created and gets updates
                
                // Base score for measurements that are unassociated and far from existing tracks
                // This indicates they're likely from a new target
                score += 1.0; // Base score for being unassociated and separated
                
                // If we have a previous measurement AND it's close in time, we can do basic validation
                // But be lenient - the previous measurement might be from a different target
                if let Some(prev_meas) = prev_meas_ref {
                    let dt = (measurement.time - prev_meas.time).max(0.1);
                    if dt > 0.0 && dt < 2.0 { // Only use if very recent (same scan)
                        let estimated_velocity = (measurement.z - prev_meas.z) / dt;
                        let speed = estimated_velocity.magnitude();
                        
                        // Velocity gate: reasonable aircraft speeds (0-300 m/s ≈ 0-1080 km/h)
                        // But be lenient - this might be from a different target
                        if speed > 0.0 && speed < 500.0 { // More lenient upper bound
                            // Score based on how reasonable the speed is
                            if speed >= 50.0 && speed <= 250.0 {
                                score += 1.0; // Good speed range (bonus)
                            } else if speed >= 20.0 && speed < 50.0 {
                                score += 0.5; // Slow but possible
                            } else if speed > 250.0 && speed < 500.0 {
                                score += 0.5; // Fast but possible (supersonic aircraft)
                            }
                            // Don't penalize other speeds - might be from different target
                        }
                        // Don't reject based on velocity - might be from different target
                    }
                }
                
                // Account for clutter density
                let clutter_penalty = jpda_lambda_fa * 1e6; // Approximate volume penalty
                score -= clutter_penalty.ln().max(0.0) * 0.1;
                
                // Detection probability bonus
                score += jpda_pd.ln() * 0.5;
                
                if score >= min_track_score {
                    log::debug!("[TRACK INIT] Measurement {} at ({:.1}, {:.1}, {:.1}) has score={:.2} >= {:.2}", 
                        meas_idx, measurement.z[0], measurement.z[1], measurement.z[2], score, min_track_score);
                    Some((measurement.clone(), score))
                } else {
                    log::debug!("[TRACK INIT] Skipping measurement {} - score={:.2} < {:.2} (likely clutter)", 
                        meas_idx, score, min_track_score);
                    None
                }
            })
            .collect();
        
        // Initialize tracks from high-quality measurements only
        // Sort by score (highest first) to prioritize best candidates
        let mut sorted_measurements = measurements_to_initialize;
        sorted_measurements.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit number of new tracks per step to prevent explosion
        // But allow enough to create all expected targets
        let max_new_tracks_per_step = match self.expected_num_targets {
            1..=2 => 3,  // Allow 3 new tracks per step for 1-2 targets (some margin)
            3..=5 => 5,  // Allow 5 for 3-5 targets
            _ => 10,     // Allow 10 for 6+ targets
        };
        
        // Also limit by how many tracks we still need
        let tracks_needed = self.expected_num_targets.saturating_sub(self.tracks.len());
        let num_to_initialize = sorted_measurements.len()
            .min(max_new_tracks_per_step)
            .min(tracks_needed.max(1)); // Always allow at least 1 if available
        
        for (measurement, score) in sorted_measurements.into_iter().take(num_to_initialize) {
            log::info!("[TRACK INIT] Creating pending track {} from measurement at ({:.1}, {:.1}, {:.1}) with score={:.2}", 
                self.next_track_id, measurement.z[0], measurement.z[1], measurement.z[2], score);
            self.initialize_track(&measurement, time);
        }
        
        if num_to_initialize > 0 {
            log::info!("[TRACK INIT] Now have {} tracks total ({} pending, {} confirmed)", 
                self.tracks.len(), 
                self.tracks.iter().filter(|t| !t.is_confirmed).count(),
                self.tracks.iter().filter(|t| t.is_confirmed).count());
        }
    }
    
    
    /// Delete tracks that should be deleted
    /// 
    /// State-of-the-art deletion logic:
    /// - More aggressive deletion for pending (unconfirmed) tracks
    /// - Lenient deletion for confirmed tracks
    /// - Consider track quality (existence probability, missed detections)
    fn delete_tracks(&mut self) {
        let mut to_delete = Vec::new();
        
        for (idx, track) in self.tracks.iter().enumerate() {
            // More aggressive deletion for pending tracks (filter out poor ones early)
            if !track.is_confirmed {
                // Pending tracks: delete if:
                // - Too many missed detections (5 for pending vs 20 for confirmed)
                // - Very low existence probability (< 0.05 for pending vs 0.01 for confirmed)
                // - Old enough (age > 10) but still not confirmed
                let should_delete_pending = 
                    track.missed_detections >= 5 ||
                    track.existence_prob < 0.05 ||
                    (track.age > 10 && track.num_detections < 2);
                
                if should_delete_pending {
                    log::info!("[TRACK DELETE] Marking pending track {} for deletion (age={}, missed={}, detections={}, existence={:.3})", 
                        track.id, track.age, track.missed_detections, track.num_detections, track.existence_prob);
                    to_delete.push(idx);
                }
            } else {
                // Confirmed tracks: use standard deletion criteria
                if track.should_delete(self.max_missed_detections, self.min_existence_prob) {
                    log::info!("[TRACK DELETE] Marking confirmed track {} for deletion (age={}, missed={}, existence={:.3})", 
                        track.id, track.age, track.missed_detections, track.existence_prob);
                    to_delete.push(idx);
                }
            }
        }
        
        if !to_delete.is_empty() {
            log::info!("[TRACK DELETE] Deleting {} tracks (had {} total)", to_delete.len(), self.tracks.len());
        }
        
        // Delete in reverse order to maintain indices
        for &idx in to_delete.iter().rev() {
            let track_id = self.tracks[idx].id;
            self.tracks.remove(idx);
            self.imm_filters.remove(idx);
            self.model_states.remove(idx);
            log::info!("[TRACK DELETE] Deleted track {}", track_id);
        }
        
        if !to_delete.is_empty() {
            log::info!("[TRACK DELETE] Now have {} tracks remaining", self.tracks.len());
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
        let tracker = Tracker::new(1.0, 1);
        assert_eq!(tracker.tracks.len(), 0);
    }
    
    #[test]
    fn test_track_initialization() {
        let mut tracker = Tracker::new(1.0, 1);
        let measurement = Measurement::with_default_covariance(
            nalgebra::Vector3::new(100.0, 200.0, 300.0),
            0.0,
        );
        
        tracker.update(&[measurement], 0.0);
        
        assert_eq!(tracker.tracks.len(), 1);
    }
    
    #[test]
    fn test_track_update() {
        let mut tracker = Tracker::new(1.0, 1);
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
        let mut tracker = Tracker::new(1.0, 1);
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
        let mut tracker = Tracker::new(1.0, 1);
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

