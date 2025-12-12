//! State representation and measurement models
//! 
//! This module defines the core data structures for representing
//! target states, measurements, and tracks in the tracking system.

use nalgebra::{Vector3, Vector6, Matrix3, Matrix6};

/// Target state vector
/// 
/// For 3D tracking, we use a 6D state: [x, y, z, vx, vy, vz]
/// representing position and velocity in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct State {
    /// Position and velocity vector (6D: x, y, z, vx, vy, vz)
    pub x: Vector6<f64>,
    /// State covariance matrix (6x6)
    pub P: Matrix6<f64>,
}

impl State {
    /// Create a new state with given mean and covariance
    pub fn new(x: Vector6<f64>, P: Matrix6<f64>) -> Self {
        Self { x, P }
    }
    
    /// Get position as 3D vector
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.x[0], self.x[1], self.x[2])
    }
    
    /// Get velocity as 3D vector
    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.x[3], self.x[4], self.x[5])
    }
    
    /// Create zero state with large initial covariance
    pub fn zero() -> Self {
        let x = Vector6::zeros();
        let P = Matrix6::identity() * 1000.0; // Large initial uncertainty
        Self { x, P }
    }
}

/// Measurement vector
/// 
/// Measurements are assumed to be direct position measurements (x, y, z).
/// In reality, sensors often provide range/azimuth/elevation, which would
/// require a nonlinear measurement model (e.g., Extended Kalman Filter).
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    /// Position measurement (3D: x, y, z)
    pub z: Vector3<f64>,
    /// Measurement covariance matrix (3x3)
    pub R: Matrix3<f64>,
    /// Timestamp (for temporal association)
    pub time: f64,
}

impl Measurement {
    /// Create a new measurement
    pub fn new(z: Vector3<f64>, R: Matrix3<f64>, time: f64) -> Self {
        Self { z, R, time }
    }
    
    /// Create measurement with default covariance
    pub fn with_default_covariance(z: Vector3<f64>, time: f64) -> Self {
        // Default measurement noise: 10m standard deviation in each dimension
        let R = Matrix3::identity() * 100.0; // 10^2 = 100
        Self::new(z, R, time)
    }
}

/// Motion model type
/// 
/// Different motion models are used in the IMM algorithm to handle
/// different types of target motion (constant velocity, acceleration, turns).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotionModel {
    /// Constant Velocity (CV) - straight line motion
    ConstantVelocity,
    /// Constant Acceleration (CA) - accelerating motion
    ConstantAcceleration,
    /// Coordinated Turn (CT) - turning motion with constant turn rate
    CoordinatedTurn,
}

impl MotionModel {
    /// Get all available motion models
    pub fn all() -> Vec<Self> {
        vec![
            Self::ConstantVelocity,
            Self::ConstantAcceleration,
            Self::CoordinatedTurn,
        ]
    }
    
    /// Get model index for array indexing
    pub fn index(&self) -> usize {
        match self {
            Self::ConstantVelocity => 0,
            Self::ConstantAcceleration => 1,
            Self::CoordinatedTurn => 2,
        }
    }
}

/// Track structure
/// 
/// A track represents a hypothesis about a target's state and motion.
/// It contains the state estimate, model probabilities (for IMM), and
/// track quality metrics.
#[derive(Debug, Clone)]
pub struct Track {
    /// Track ID (unique identifier)
    pub id: usize,
    /// Current state estimate
    pub state: State,
    /// Model probabilities for IMM (one per motion model)
    pub model_probs: Vec<f64>,
    /// Age of the track (number of updates)
    pub age: usize,
    /// Number of consecutive missed detections
    pub missed_detections: usize,
    /// Track existence probability (for track quality)
    pub existence_prob: f64,
    /// Last update time
    pub last_update_time: f64,
    /// Track confirmation status (M/N logic: M detections out of N scans)
    pub is_confirmed: bool,
    /// Number of detections received (for M/N confirmation)
    pub num_detections: usize,
    /// Number of scans since track creation (for M/N confirmation)
    pub num_scans: usize,
    /// Measurement history for velocity estimation (last 3 measurements)
    pub measurement_history: Vec<(Vector3<f64>, f64)>, // (position, time)
}

impl Track {
    /// Create a new track
    pub fn new(id: usize, initial_state: State, num_models: usize, time: f64) -> Self {
        // Initialize with uniform model probabilities
        let model_probs = vec![1.0 / num_models as f64; num_models];
        
        Self {
            id,
            state: initial_state,
            model_probs,
            age: 0,
            missed_detections: 0,
            existence_prob: 0.5, // Initial existence probability
            last_update_time: time,
            is_confirmed: false, // Start as tentative
            num_detections: 0,
            num_scans: 0,
            measurement_history: Vec::new(),
        }
    }
    
    /// Check if track should be confirmed (M/N logic)
    /// M detections out of N scans required for confirmation
    pub fn should_confirm(&self, m: usize, n: usize) -> bool {
        if self.is_confirmed {
            return true; // Already confirmed
        }
        // Require at least M detections in the last N scans
        self.num_detections >= m && self.num_scans >= n
    }
    
    /// Add a measurement to history (for velocity estimation)
    pub fn add_measurement(&mut self, position: Vector3<f64>, time: f64) {
        self.measurement_history.push((position, time));
        // Keep only last 3 measurements
        if self.measurement_history.len() > 3 {
            self.measurement_history.remove(0);
        }
        self.num_detections += 1;
    }
    
    /// Increment scan counter
    pub fn increment_scan(&mut self) {
        self.num_scans += 1;
    }
    
    /// Check if track should be deleted
    /// 
    /// Conservative deletion logic: require both conditions AND track must be old enough
    /// to prevent premature deletion of valid tracks.
    pub fn should_delete(&self, max_missed: usize, min_existence: f64) -> bool {
        // Don't delete very young tracks (less than 5 updates) - they're still initializing
        if self.age < 5 {
            return false;
        }
        
        // For older tracks, require BOTH conditions to be true
        // This prevents deletion due to temporary issues
        let too_many_missed = self.missed_detections >= max_missed;
        let existence_too_low = self.existence_prob < min_existence;
        
        // Only delete if BOTH conditions are met AND track is reasonably old
        if self.age >= 10 {
            too_many_missed && existence_too_low
        } else {
            // For tracks between 5-10 updates, be even more conservative
            // Require existence to be very low AND many missed detections
            self.existence_prob < min_existence * 0.5 && self.missed_detections >= max_missed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_creation() {
        let x = Vector6::new(100.0, 200.0, 300.0, 10.0, 20.0, 30.0);
        let P = Matrix6::identity() * 50.0;
        let state = State::new(x, P);
        
        assert_eq!(state.position(), Vector3::new(100.0, 200.0, 300.0));
        assert_eq!(state.velocity(), Vector3::new(10.0, 20.0, 30.0));
    }
    
    #[test]
    fn test_measurement_creation() {
        let z = Vector3::new(100.0, 200.0, 300.0);
        let measurement = Measurement::with_default_covariance(z, 0.0);
        
        assert_eq!(measurement.z, z);
        assert_eq!(measurement.time, 0.0);
    }
    
    #[test]
    fn test_track_creation() {
        let state = State::zero();
        let track = Track::new(0, state, 3, 0.0);
        
        assert_eq!(track.id, 0);
        assert_eq!(track.model_probs.len(), 3);
        assert!((track.model_probs[0] - 1.0/3.0).abs() < 1e-10);
        assert_eq!(track.age, 0);
    }
    
    #[test]
    fn test_track_deletion() {
        let state = State::zero();
        let mut track = Track::new(0, state, 3, 0.0);
        
        // Should not be deleted initially
        assert!(!track.should_delete(5, 0.1));
        
        // Should be deleted after too many missed detections
        track.missed_detections = 5;
        assert!(track.should_delete(5, 0.1));
        
        // Should be deleted if existence probability too low
        track.missed_detections = 0;
        track.existence_prob = 0.05;
        assert!(track.should_delete(5, 0.1));
    }
}

