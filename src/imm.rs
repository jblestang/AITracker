//! Interacting Multiple Model (IMM) Algorithm
//! 
//! The IMM algorithm manages multiple Kalman filters running in parallel,
//! each representing a different motion model. It probabilistically switches
//! between models based on measurement likelihoods.
//! 
//! Algorithm steps:
//! 1. Mixing: Combine model-conditioned estimates based on transition probabilities
//! 2. Prediction: Predict each filter forward
//! 3. Update: Update each filter with measurements
//! 4. Model probability update: Update model probabilities based on likelihoods
//! 5. Combination: Combine estimates weighted by model probabilities

use nalgebra::Vector6;
use crate::state::{State, Measurement, MotionModel};
use crate::kalman::{KalmanFilter, get_filter};

/// IMM algorithm implementation
pub struct IMM {
    /// Motion models used in IMM
    models: Vec<MotionModel>,
    /// Model transition probability matrix (Markov chain)
    /// transition[i][j] = P(model_j at k+1 | model_i at k)
    transition_matrix: Vec<Vec<f64>>,
    /// Current model probabilities
    model_probs: Vec<f64>,
    /// Filters for each model
    filters: Vec<Box<dyn KalmanFilter>>,
}

impl IMM {
    /// Create a new IMM filter
    /// 
    /// # Arguments
    /// * `models` - List of motion models to use
    /// * `initial_probs` - Initial model probabilities (must sum to 1.0)
    /// * `transition_matrix` - Model transition probability matrix
    pub fn new(
        models: Vec<MotionModel>,
        initial_probs: Vec<f64>,
        transition_matrix: Vec<Vec<f64>>,
    ) -> Self {
        assert_eq!(models.len(), initial_probs.len());
        assert_eq!(models.len(), transition_matrix.len());
        
        // Normalize initial probabilities
        let sum: f64 = initial_probs.iter().sum();
        let model_probs: Vec<f64> = initial_probs.iter().map(|&p| p / sum).collect();
        
        // Create filters for each model
        let filters: Vec<Box<dyn KalmanFilter>> = models.iter()
            .map(|&model| get_filter(model))
            .collect();
        
        Self {
            models,
            transition_matrix,
            model_probs,
            filters,
        }
    }
    
    /// Create IMM with default parameters
    /// 
    /// Uses all three motion models with uniform initial probabilities
    /// and a simple transition matrix.
    pub fn default() -> Self {
        let models = MotionModel::all();
        let num_models = models.len();
        
        // Uniform initial probabilities
        let initial_probs = vec![1.0 / num_models as f64; num_models];
        
        // Adaptive transition matrix: high probability of staying in same model,
        // but allow more switching between CV and CA/CT based on motion characteristics
        let mut transition_matrix = vec![vec![0.0; num_models]; num_models];
        for i in 0..num_models {
            for j in 0..num_models {
                if i == j {
                    transition_matrix[i][j] = 0.90; // Stay in same model (slightly lower for more adaptability)
                } else {
                    // Prefer transitions between similar models (CV <-> CA, CA <-> CT)
                    // CV (0) <-> CA (1) is more likely than CV <-> CT
                    if (i == 0 && j == 1) || (i == 1 && j == 0) {
                        transition_matrix[i][j] = 0.05; // CV <-> CA
                    } else if (i == 1 && j == 2) || (i == 2 && j == 1) {
                        transition_matrix[i][j] = 0.05; // CA <-> CT
                    } else {
                        transition_matrix[i][j] = 0.025; // CV <-> CT (less likely)
                    }
                }
            }
        }
        
        Self::new(models, initial_probs, transition_matrix)
    }
    
    /// Mix model-conditioned estimates
    /// 
    /// Computes mixed initial conditions for each filter based on
    /// model transition probabilities and previous model probabilities.
    /// 
    /// # Arguments
    /// * `states` - Model-conditioned state estimates from previous time
    /// 
    /// # Returns
    /// Mixed initial states for each model
    fn mix(&self, states: &[State]) -> Vec<State> {
        let num_models = self.models.len();
        let mut mixed_states = Vec::new();
        
        // Compute mixing probabilities: mu(i|j) = P(model_i at k | model_j at k+1)
        // Using Bayes' rule: mu(i|j) = transition[i][j] * model_prob[i] / c_j
        // where c_j is normalization constant
        let mut mixing_probs = vec![vec![0.0; num_models]; num_models];
        
        for j in 0..num_models {
            // Normalization constant for model j
            let mut c_j = 0.0;
            for i in 0..num_models {
                c_j += self.transition_matrix[i][j] * self.model_probs[i];
            }
            
            // Compute mixing probabilities
            if c_j > 1e-10 {
                for i in 0..num_models {
                    mixing_probs[i][j] = self.transition_matrix[i][j] * self.model_probs[i] / c_j;
                }
            } else {
                // Fallback to uniform if normalization too small
                for i in 0..num_models {
                    mixing_probs[i][j] = 1.0 / num_models as f64;
                }
            }
        }
        
        // Mix states for each model
        for j in 0..num_models {
            let mut mixed_x = Vector6::zeros();
            let mut mixed_P = nalgebra::Matrix6::zeros();
            
            // Weighted sum of states
            for i in 0..num_models {
                let weight = mixing_probs[i][j];
                mixed_x += weight * states[i].x;
            }
            
            // Mix covariance (accounting for spread of means)
            for i in 0..num_models {
                let weight = mixing_probs[i][j];
                let dx = states[i].x - mixed_x;
                mixed_P += weight * (states[i].P + dx * dx.transpose());
            }
            
            mixed_states.push(State::new(mixed_x, mixed_P));
        }
        
        mixed_states
    }
    
    /// Predict all filters forward in time
    /// 
    /// # Arguments
    /// * `states` - Current state estimates for each model
    /// * `dt` - Time step
    /// 
    /// # Returns
    /// Predicted states for each model
    fn predict_all(&self, states: &[State], dt: f64) -> Vec<State> {
        states.iter()
            .zip(self.filters.iter())
            .map(|(state, filter)| filter.predict(state, dt))
            .collect()
    }
    
    /// Update all filters with measurement
    /// 
    /// # Arguments
    /// * `states` - Predicted states for each model
    /// * `measurement` - New measurement
    /// 
    /// # Returns
    /// Updated states and likelihoods for each model
    fn update_all(&self, states: &[State], measurement: &Measurement) -> (Vec<State>, Vec<f64>) {
        let mut updated_states = Vec::new();
        let mut likelihoods = Vec::new();
        
        for (state, filter) in states.iter().zip(self.filters.iter()) {
            let (updated, _) = filter.update(state, measurement);
            let likelihood = filter.likelihood(state, measurement);
            
            updated_states.push(updated);
            likelihoods.push(likelihood);
        }
        
        (updated_states, likelihoods)
    }
    
    /// Update model probabilities based on measurement likelihoods
    /// 
    /// # Arguments
    /// * `likelihoods` - Measurement likelihoods for each model
    /// 
    /// # Returns
    /// Updated model probabilities
    fn update_model_probs(&mut self, likelihoods: &[f64]) {
        let num_models = self.models.len();
        
        // Compute normalization constant
        let mut c = 0.0;
        for j in 0..num_models {
            let mut sum = 0.0;
            for i in 0..num_models {
                sum += self.transition_matrix[i][j] * self.model_probs[i];
            }
            c += likelihoods[j] * sum;
        }
        
        // Update model probabilities
        if c > 1e-10 {
            for j in 0..num_models {
                let mut sum = 0.0;
                for i in 0..num_models {
                    sum += self.transition_matrix[i][j] * self.model_probs[i];
                }
                self.model_probs[j] = likelihoods[j] * sum / c;
            }
        }
        
        // Normalize to ensure probabilities sum to 1
        let sum: f64 = self.model_probs.iter().sum();
        if sum > 1e-10 {
            for prob in &mut self.model_probs {
                *prob /= sum;
            }
        }
    }
    
    /// Combine model-conditioned estimates
    /// 
    /// Computes overall state estimate as weighted combination of
    /// model-conditioned estimates, weighted by model probabilities.
    /// 
    /// # Arguments
    /// * `states` - Model-conditioned state estimates
    /// 
    /// # Returns
    /// Combined state estimate
    fn combine(&self, states: &[State]) -> State {
        
        // Weighted mean
        let mut combined_x = Vector6::zeros();
        for (state, &prob) in states.iter().zip(self.model_probs.iter()) {
            combined_x += prob * state.x;
        }
        
        // Combined covariance (accounting for spread of means)
        let mut combined_P = nalgebra::Matrix6::zeros();
        for (state, &prob) in states.iter().zip(self.model_probs.iter()) {
            let dx = state.x - combined_x;
            combined_P += prob * (state.P + dx * dx.transpose());
        }
        
        State::new(combined_x, combined_P)
    }
    
    /// Run one IMM cycle: predict and update
    /// 
    /// # Arguments
    /// * `states` - Current model-conditioned state estimates
    /// * `measurement` - New measurement (None for prediction only)
    /// * `dt` - Time step
    /// 
    /// # Returns
    /// Updated model-conditioned states, combined state, and updated model probabilities
    pub fn step(
        &mut self,
        states: &[State],
        measurement: Option<&Measurement>,
        dt: f64,
    ) -> (Vec<State>, State, Vec<f64>) {
        // Step 1: Mixing
        let mixed_states = self.mix(states);
        
        // Step 2: Prediction
        let predicted_states = self.predict_all(&mixed_states, dt);
        
        // Step 3: Update (if measurement available)
        let (updated_states, likelihoods) = if let Some(meas) = measurement {
            self.update_all(&predicted_states, meas)
        } else {
            // No measurement - use predicted states and zero likelihoods
            (predicted_states.clone(), vec![1.0; self.models.len()])
        };
        
        // Step 4: Update model probabilities
        if measurement.is_some() {
            self.update_model_probs(&likelihoods);
        }
        
        // Step 5: Combine
        let combined_state = self.combine(&updated_states);
        
        (updated_states, combined_state, self.model_probs.clone())
    }
    
    /// Get current model probabilities
    pub fn model_probs(&self) -> &[f64] {
        &self.model_probs
    }
    
    /// Set model probabilities (for reconstruction during parallel processing)
    pub fn set_model_probs(&mut self, probs: &[f64]) {
        let sum: f64 = probs.iter().sum();
        if sum > 1e-10 {
            self.model_probs = probs.iter().map(|&p| p / sum).collect();
        }
    }
    
    /// Update model probabilities directly (public method for external control)
    pub fn update_model_probs_direct(&mut self, likelihoods: &[f64]) {
        self.update_model_probs(likelihoods);
    }
    
    /// Combine states (public method for external use)
    pub fn combine_states(&self, states: &[State]) -> State {
        self.combine(states)
    }
    
    /// Mix and predict only (for use with external updates)
    pub fn mix_and_predict(&self, states: &[State], dt: f64) -> Vec<State> {
        let mixed_states = self.mix(states);
        self.predict_all(&mixed_states, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_imm_creation() {
        let imm = IMM::default();
        let probs = imm.model_probs();
        
        // Should have 3 models
        assert_eq!(probs.len(), 3);
        
        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_imm_mixing() {
        let imm = IMM::default();
        let states = vec![
            State::zero(),
            State::zero(),
            State::zero(),
        ];
        
        let mixed = imm.mix(&states);
        
        // Should have same number of mixed states as models
        assert_eq!(mixed.len(), 3);
    }
    
    #[test]
    fn test_imm_prediction() {
        let imm = IMM::default();
        let x = Vector6::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let P = nalgebra::Matrix6::identity() * 10.0;
        let states = vec![State::new(x, P); 3];
        
        let predicted = imm.predict_all(&states, 1.0);
        
        // Should predict forward
        assert_eq!(predicted.len(), 3);
        assert!(predicted[0].x[0] > 0.0);
    }
    
    #[test]
    fn test_imm_update() {
        let imm = IMM::default();
        let x = Vector6::new(100.0, 200.0, 300.0, 0.0, 0.0, 0.0);
        let P = nalgebra::Matrix6::identity() * 10.0;
        let states = vec![State::new(x, P); 3];
        
        let measurement = Measurement::new(
            nalgebra::Vector3::new(110.0, 210.0, 310.0),
            nalgebra::Matrix3::identity() * 10.0,
            0.0,
        );
        
        let (updated, likelihoods) = imm.update_all(&states, &measurement);
        
        assert_eq!(updated.len(), 3);
        assert_eq!(likelihoods.len(), 3);
        
        // All likelihoods should be positive
        for &likelihood in &likelihoods {
            assert!(likelihood > 0.0);
        }
    }
    
    #[test]
    fn test_imm_full_step() {
        let mut imm = IMM::default();
        let x = Vector6::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let P = nalgebra::Matrix6::identity() * 10.0;
        let states = vec![State::new(x, P); 3];
        
        let measurement = Measurement::new(
            nalgebra::Vector3::new(10.0, 20.0, 30.0),
            nalgebra::Matrix3::identity() * 10.0,
            1.0,
        );
        
        let (updated_states, combined, probs) = imm.step(&states, Some(&measurement), 1.0);
        
        assert_eq!(updated_states.len(), 3);
        assert_eq!(probs.len(), 3);
        
        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Combined state should be reasonable
        assert!(combined.x[0].abs() < 1000.0);
    }
    
    #[test]
    fn test_model_probability_update() {
        let mut imm = IMM::default();
        
        // High likelihood for first model, low for others
        let likelihoods = vec![1.0, 0.1, 0.1];
        
        imm.update_model_probs(&likelihoods);
        let probs = imm.model_probs();
        
        // First model should have higher probability
        assert!(probs[0] > probs[1]);
        assert!(probs[0] > probs[2]);
        
        // Should still sum to 1
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }
}

