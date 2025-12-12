//! Kalman filter implementations for different motion models
//! 
//! This module implements Kalman filters for various motion models:
//! - Constant Velocity (CV)
//! - Constant Acceleration (CA)
//! - Coordinated Turn (CT)
//! 
//! Each filter implements prediction and update steps for state estimation.

use nalgebra::{Vector3, Vector6, Matrix3, Matrix6, Matrix3x6};
use crate::state::{State, Measurement, MotionModel};

/// Measurement matrix (H) - maps state to measurement space
/// 
/// Since we measure position directly, H = [I_3x3, 0_3x3]
/// This extracts position from the 6D state vector.
const H: Matrix3x6<f64> = Matrix3x6::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
);

/// Kalman filter for Constant Velocity (CV) motion model
/// 
/// Assumes target moves with constant velocity:
/// x(k+1) = x(k) + v(k)*dt
/// v(k+1) = v(k)
pub struct CVFilter {
    /// Process noise covariance (for velocity uncertainty)
    pub Q: Matrix6<f64>,
}

impl CVFilter {
    /// Create a new CV filter
    /// 
    /// # Arguments
    /// * `process_noise` - Standard deviation of velocity process noise
    pub fn new(process_noise: f64) -> Self {
        // Process noise only affects velocity components
        // Use smaller process noise to allow velocity to converge faster
        let mut Q = Matrix6::zeros();
        let q = process_noise * process_noise;
        Q[(3, 3)] = q; // vx noise
        Q[(4, 4)] = q; // vy noise
        Q[(5, 5)] = q; // vz noise
        
        // Add small position-velocity cross-coupling in process noise
        // This helps velocity estimates converge
        let cross_q = q * 0.1;
        Q[(0, 3)] = cross_q;
        Q[(3, 0)] = cross_q;
        Q[(1, 4)] = cross_q;
        Q[(4, 1)] = cross_q;
        Q[(2, 5)] = cross_q;
        Q[(5, 2)] = cross_q;
        
        Self { Q }
    }
    
    /// Predict state forward in time
    /// 
    /// # Arguments
    /// * `state` - Current state estimate
    /// * `dt` - Time step
    /// 
    /// # Returns
    /// Predicted state and covariance
    pub fn predict(&self, state: &State, dt: f64) -> State {
        // State transition matrix for CV model
        let mut F = Matrix6::identity();
        F[(0, 3)] = dt; // x = x + vx*dt
        F[(1, 4)] = dt; // y = y + vy*dt
        F[(2, 5)] = dt; // z = z + vz*dt
        
        // Predict state: x_pred = F * x
        let x_pred = F * state.x;
        
        // Predict covariance: P_pred = F * P * F' + Q
        let P_pred = F * state.P * F.transpose() + self.Q;
        
        State::new(x_pred, P_pred)
    }
    
    /// Update state with measurement
    /// 
    /// # Arguments
    /// * `state` - Predicted state
    /// * `measurement` - New measurement
    /// 
    /// # Returns
    /// Updated state and covariance, and innovation (for likelihood computation)
    pub fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>) {
        // Innovation: y = z - H*x
        let y = measurement.z - H * state.x;
        
        // Innovation covariance: S = H*P*H' + R
        let S = H * state.P * H.transpose() + measurement.R;
        
        // Kalman gain: K = P*H'*S^(-1)
        // The gain matrix has shape 6x3, where:
        // - First 3 rows correspond to position updates
        // - Last 3 rows correspond to velocity updates (through cross-covariance)
        let K = state.P * H.transpose() * S.try_inverse().unwrap();
        
        // DEBUG: Log Kalman gain for velocity
        log::debug!("Kalman Filter Update:");
        log::debug!("  Innovation: ({:.2}, {:.2}, {:.2})", y[0], y[1], y[2]);
        log::debug!("  Velocity gain K[3,:]: ({:.4}, {:.4}, {:.4})", K[(3, 0)], K[(3, 1)], K[(3, 2)]);
        log::debug!("  Velocity gain K[4,:]: ({:.4}, {:.4}, {:.4})", K[(4, 0)], K[(4, 1)], K[(4, 2)]);
        log::debug!("  Velocity gain K[5,:]: ({:.4}, {:.4}, {:.4})", K[(5, 0)], K[(5, 1)], K[(5, 2)]);
        log::debug!("  Cross-covariance P[(0,3)]: {:.2}, P[(1,4)]: {:.2}, P[(2,5)]: {:.2}", 
            state.P[(0, 3)], state.P[(1, 4)], state.P[(2, 5)]);
        log::debug!("  Velocity before update: ({:.2}, {:.2}, {:.2})", state.x[3], state.x[4], state.x[5]);
        
        // Update state: x = x + K*y
        // This updates both position AND velocity (velocity through cross-covariance in K)
        let x_updated = state.x + K * y;
        
        log::debug!("  Velocity after K*y: ({:.2}, {:.2}, {:.2})", x_updated[3], x_updated[4], x_updated[5]);
        
        // Ensure velocity is updated: if cross-covariance is weak, velocity might not update
        // Check if velocity rows of K are significant
        let vel_gain_magnitude = (K[(3, 0)].abs() + K[(3, 1)].abs() + K[(3, 2)].abs() +
                                  K[(4, 0)].abs() + K[(4, 1)].abs() + K[(4, 2)].abs() +
                                  K[(5, 0)].abs() + K[(5, 1)].abs() + K[(5, 2)].abs()) / 9.0;
        
        log::debug!("  Velocity gain magnitude: {:.4}", vel_gain_magnitude);
        
        // If velocity gain is too small, velocity won't update properly
        // This can happen if cross-covariance is weak
        if vel_gain_magnitude < 0.01 {
            log::debug!("  WARNING: Velocity gain is too weak - cross-covariance may not be working!");
        }
        
        // Update covariance: P = (I - K*H)*P
        let I = Matrix6::identity();
        let P_updated = (I - K * H) * state.P;
        
        (State::new(x_updated, P_updated), y)
    }
    
    /// Compute measurement likelihood
    /// 
    /// Returns the probability density of the measurement given the predicted state.
    /// This is used in IMM for model probability updates.
    pub fn likelihood(&self, state: &State, measurement: &Measurement) -> f64 {
        // Innovation
        let y = measurement.z - H * state.x;
        
        // Innovation covariance
        let S = H * state.P * H.transpose() + measurement.R;
        
        // Multivariate Gaussian likelihood: exp(-0.5 * y' * S^(-1) * y) / sqrt(det(2*pi*S))
        let S_inv = match S.try_inverse() {
            Some(inv) => inv,
            None => {
                log::debug!("    Likelihood: S is singular, returning 0");
                return 0.0;
            }
        };
        let mahalanobis_sq = (y.transpose() * S_inv * y)[0];
        let exponent_val = -0.5 * mahalanobis_sq;
        let det_S = S.determinant();
        let normalization = (2.0 * std::f64::consts::PI).powf(1.5) * det_S.sqrt();
        
        let likelihood = exponent_val.exp() / normalization;
        
        log::debug!("    Likelihood computation: mahalanobis_sq={:.2}, exponent={:.2}, det_S={:.2}, norm={:.2}, likelihood={:.6e}", 
            mahalanobis_sq, exponent_val, det_S, normalization, likelihood);
        
        likelihood
    }
}

/// Kalman filter for Constant Acceleration (CA) motion model
/// 
/// Assumes target moves with constant acceleration:
/// x(k+1) = x(k) + v(k)*dt + 0.5*a(k)*dt^2
/// v(k+1) = v(k) + a(k)*dt
/// a(k+1) = a(k)
/// 
/// Note: This is simplified - we use a 6D state (position + velocity)
/// and model acceleration as process noise on velocity.
pub struct CAFilter {
    /// Process noise covariance (for acceleration uncertainty)
    pub Q: Matrix6<f64>,
}

impl CAFilter {
    /// Create a new CA filter
    /// 
    /// # Arguments
    /// * `process_noise` - Standard deviation of acceleration process noise
    pub fn new(process_noise: f64) -> Self {
        // Process noise affects velocity (modeling acceleration)
        let mut Q = Matrix6::zeros();
        let q = process_noise * process_noise;
        Q[(3, 3)] = q;
        Q[(4, 4)] = q;
        Q[(5, 5)] = q;
        
        Self { Q }
    }
    
    /// Predict state forward in time
    pub fn predict(&self, state: &State, dt: f64) -> State {
        // State transition matrix for CA model (simplified)
        // We model acceleration as process noise, so same as CV but with larger Q
        let mut F = Matrix6::identity();
        F[(0, 3)] = dt;
        F[(1, 4)] = dt;
        F[(2, 5)] = dt;
        
        let x_pred = F * state.x;
        let P_pred = F * state.P * F.transpose() + self.Q;
        
        State::new(x_pred, P_pred)
    }
    
    /// Update state with measurement
    pub fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>) {
        let y = measurement.z - H * state.x;
        let S = H * state.P * H.transpose() + measurement.R;
        let K = state.P * H.transpose() * S.try_inverse().unwrap();
        let x_updated = state.x + K * y;
        let I = Matrix6::identity();
        let P_updated = (I - K * H) * state.P;
        
        (State::new(x_updated, P_updated), y)
    }
    
    /// Compute measurement likelihood
    pub fn likelihood(&self, state: &State, measurement: &Measurement) -> f64 {
        let y = measurement.z - H * state.x;
        let S = H * state.P * H.transpose() + measurement.R;
        let S_inv = S.try_inverse().unwrap();
        let exponent_val = -0.5 * (y.transpose() * S_inv * y)[0];
        let det_S = S.determinant();
        let normalization = (2.0 * std::f64::consts::PI).powf(1.5) * det_S.sqrt();
        
        exponent_val.exp() / normalization
    }
}

/// Kalman filter for Coordinated Turn (CT) motion model
/// 
/// Assumes target moves in a coordinated turn with constant turn rate.
/// This is a simplified linear approximation - in reality, CT requires
/// an Extended Kalman Filter (EKF) due to nonlinear dynamics.
/// 
/// For simplicity, we use a linearized version around zero turn rate.
pub struct CTFilter {
    /// Process noise covariance
    pub Q: Matrix6<f64>,
    /// Turn rate (rad/s) - fixed for simplicity
    pub turn_rate: f64,
}

impl CTFilter {
    /// Create a new CT filter
    /// 
    /// # Arguments
    /// * `process_noise` - Standard deviation of process noise
    /// * `turn_rate` - Turn rate in rad/s (simplified - fixed value)
    pub fn new(process_noise: f64, turn_rate: f64) -> Self {
        let mut Q = Matrix6::zeros();
        let q = process_noise * process_noise;
        Q[(3, 3)] = q;
        Q[(4, 4)] = q;
        Q[(5, 5)] = q;
        
        Self { Q, turn_rate }
    }
    
    /// Predict state forward in time
    /// 
    /// Simplified linear approximation of coordinated turn.
    /// In reality, this requires EKF with nonlinear state transition.
    pub fn predict(&self, state: &State, dt: f64) -> State {
        // Linearized CT model (simplified)
        // For small turn rates, we approximate as CV with cross-coupling
        let mut F = Matrix6::identity();
        F[(0, 3)] = dt;
        F[(1, 4)] = dt;
        F[(2, 5)] = dt;
        
        // Add turn coupling (simplified linear approximation)
        if self.turn_rate.abs() > 1e-6 {
            let omega = self.turn_rate;
            F[(3, 4)] = omega * dt; // vx coupling from vy
            F[(4, 3)] = -omega * dt; // vy coupling from vx
        }
        
        let x_pred = F * state.x;
        let P_pred = F * state.P * F.transpose() + self.Q;
        
        State::new(x_pred, P_pred)
    }
    
    /// Update state with measurement
    pub fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>) {
        let y = measurement.z - H * state.x;
        let S = H * state.P * H.transpose() + measurement.R;
        let K = state.P * H.transpose() * S.try_inverse().unwrap();
        let x_updated = state.x + K * y;
        let I = Matrix6::identity();
        let P_updated = (I - K * H) * state.P;
        
        (State::new(x_updated, P_updated), y)
    }
    
    /// Compute measurement likelihood
    pub fn likelihood(&self, state: &State, measurement: &Measurement) -> f64 {
        let y = measurement.z - H * state.x;
        let S = H * state.P * H.transpose() + measurement.R;
        let S_inv = S.try_inverse().unwrap();
        let exponent_val = -0.5 * (y.transpose() * S_inv * y)[0];
        let det_S = S.determinant();
        let normalization = (2.0 * std::f64::consts::PI).powf(1.5) * det_S.sqrt();
        
        exponent_val.exp() / normalization
    }
}

/// Get appropriate filter for a motion model
pub fn get_filter(model: MotionModel) -> Box<dyn KalmanFilter> {
    match model {
        MotionModel::ConstantVelocity => Box::new(CVFilter::new(2.0)),
        MotionModel::ConstantAcceleration => Box::new(CAFilter::new(5.0)),
        MotionModel::CoordinatedTurn => Box::new(CTFilter::new(3.0, 0.1)),
    }
}

/// Trait for Kalman filter operations
pub trait KalmanFilter {
    /// Predict state forward
    fn predict(&self, state: &State, dt: f64) -> State;
    
    /// Update state with measurement
    fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>);
    
    /// Compute measurement likelihood
    fn likelihood(&self, state: &State, measurement: &Measurement) -> f64;
}

// Implement trait for all filter types
impl KalmanFilter for CVFilter {
    fn predict(&self, state: &State, dt: f64) -> State {
        self.predict(state, dt)
    }
    
    fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>) {
        self.update(state, measurement)
    }
    
    fn likelihood(&self, state: &State, measurement: &Measurement) -> f64 {
        self.likelihood(state, measurement)
    }
}

impl KalmanFilter for CAFilter {
    fn predict(&self, state: &State, dt: f64) -> State {
        self.predict(state, dt)
    }
    
    fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>) {
        self.update(state, measurement)
    }
    
    fn likelihood(&self, state: &State, measurement: &Measurement) -> f64 {
        self.likelihood(state, measurement)
    }
}

impl KalmanFilter for CTFilter {
    fn predict(&self, state: &State, dt: f64) -> State {
        self.predict(state, dt)
    }
    
    fn update(&self, state: &State, measurement: &Measurement) -> (State, Vector3<f64>) {
        self.update(state, measurement)
    }
    
    fn likelihood(&self, state: &State, measurement: &Measurement) -> f64 {
        self.likelihood(state, measurement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_cv_filter_predict() {
        let filter = CVFilter::new(1.0);
        let x = Vector6::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let P = Matrix6::identity() * 10.0;
        let state = State::new(x, P);
        
        let predicted = filter.predict(&state, 1.0);
        
        // After 1 second, position should be (10, 20, 30)
        assert_relative_eq!(predicted.x[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(predicted.x[1], 20.0, epsilon = 1e-6);
        assert_relative_eq!(predicted.x[2], 30.0, epsilon = 1e-6);
        // Velocity should remain the same
        assert_relative_eq!(predicted.x[3], 10.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_cv_filter_update() {
        let filter = CVFilter::new(1.0);
        let x = Vector6::new(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let P = Matrix6::identity() * 100.0;
        let state = State::new(x, P);
        
        let measurement = Measurement::new(
            Vector3::new(110.0, 210.0, 310.0),
            Matrix3::identity() * 10.0,
            0.0,
        );
        
        let (updated, innovation) = filter.update(&state, &measurement);
        
        // Innovation should be (10, 10, 10)
        assert_relative_eq!(innovation[0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(innovation[1], 10.0, epsilon = 1e-6);
        assert_relative_eq!(innovation[2], 10.0, epsilon = 1e-6);
        
        // Updated state should be between prediction and measurement
        assert!(updated.x[0] > 100.0 && updated.x[0] < 110.0);
    }
    
    #[test]
    fn test_likelihood_computation() {
        let filter = CVFilter::new(1.0);
        let x = Vector6::new(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let P = Matrix6::identity() * 10.0;
        let state = State::new(x, P);
        
        // Measurement exactly at predicted position
        let measurement = Measurement::new(
            Vector3::new(100.0, 200.0, 300.0),
            Matrix3::identity() * 10.0,
            0.0,
        );
        
        let likelihood = filter.likelihood(&state, &measurement);
        
        // Likelihood should be positive
        assert!(likelihood > 0.0);
        assert!(likelihood < 1.0);
        
        // Measurement far away should have lower likelihood
        let far_measurement = Measurement::new(
            Vector3::new(200.0, 300.0, 400.0),
            Matrix3::identity() * 10.0,
            0.0,
        );
        let far_likelihood = filter.likelihood(&state, &far_measurement);
        assert!(far_likelihood < likelihood);
    }
    
    #[test]
    fn test_ca_filter() {
        let filter = CAFilter::new(2.0);
        let x = Vector6::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let P = Matrix6::identity() * 10.0;
        let state = State::new(x, P);
        
        let predicted = filter.predict(&state, 1.0);
        
        // Should behave similarly to CV for prediction
        assert_relative_eq!(predicted.x[0], 10.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_ct_filter() {
        let filter = CTFilter::new(2.0, 0.1);
        let x = Vector6::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let P = Matrix6::identity() * 10.0;
        let state = State::new(x, P);
        
        let predicted = filter.predict(&state, 1.0);
        
        // Should predict forward
        assert!(predicted.x[0].abs() > 0.0 || predicted.x[1].abs() > 0.0);
    }
}

