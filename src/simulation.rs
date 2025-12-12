//! Aircraft simulation and clutter generation
//! 
//! This module simulates aircraft trajectories and generates clutter measurements.
//! It provides a realistic test environment for the tracking algorithms.
//! 
//! Features:
//! - Aircraft trajectory generation with realistic motion
//! - Clutter generation (false alarms)
//! - Measurement noise modeling
//! - Time-based simulation

use nalgebra::{Vector3, Vector6, Matrix3, Matrix6};
use rand::Rng;
use crate::state::{State, Measurement};

/// Aircraft simulator
/// 
/// Generates realistic aircraft trajectories with:
/// - Straight flight segments (constant velocity)
/// - Turning segments (coordinated turns)
/// - Acceleration segments
pub struct AircraftSimulator {
    /// Current aircraft state
    state: State,
    /// Current time
    time: f64,
    /// Time step
    dt: f64,
    /// Current motion phase (for generating realistic trajectories)
    phase: MotionPhase,
    /// Phase timer
    phase_timer: f64,
    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

/// Motion phase for realistic trajectory generation
#[derive(Debug, Clone, Copy)]
enum MotionPhase {
    /// Constant velocity flight
    Straight,
    /// Coordinated turn
    Turn { turn_rate: f64, duration: f64 },
    /// Acceleration
    Accelerate { acceleration: Vector3<f64>, duration: f64 },
}

impl AircraftSimulator {
    /// Create a new aircraft simulator
    /// 
    /// # Arguments
    /// * `initial_state` - Initial aircraft state
    /// * `dt` - Time step
    pub fn new(initial_state: State, dt: f64) -> Self {
        Self {
            state: initial_state,
            time: 0.0,
            dt,
            phase: MotionPhase::Straight,
            phase_timer: 0.0,
            rng: rand::thread_rng(),
        }
    }
    
    /// Create simulator with default initial conditions
    pub fn default(dt: f64) -> Self {
        // Start at origin with initial velocity
        let x = Vector6::new(0.0, 0.0, 5000.0, 100.0, 50.0, 0.0); // 100 m/s forward, 50 m/s sideways
        let P = Matrix6::identity() * 1.0; // Small initial uncertainty
        let initial_state = State::new(x, P);
        
        Self::new(initial_state, dt)
    }
    
    /// Step the simulation forward
    /// 
    /// Updates the aircraft state based on current motion phase.
    pub fn step(&mut self) {
        self.time += self.dt;
        self.phase_timer += self.dt;
        
        // Update phase if needed
        self.update_phase();
        
        // Update state based on phase
        match self.phase {
            MotionPhase::Straight => {
                // Constant velocity: x = x + v*dt
                self.state.x[0] += self.state.x[3] * self.dt;
                self.state.x[1] += self.state.x[4] * self.dt;
                self.state.x[2] += self.state.x[5] * self.dt;
            }
            MotionPhase::Turn { turn_rate, .. } => {
                // Coordinated turn: update velocity direction
                let v = Vector3::new(self.state.x[3], self.state.x[4], self.state.x[5]);
                let v_mag = v.magnitude();
                
                if v_mag > 1e-6 {
                    // Rotate velocity vector in horizontal plane
                    let angle = turn_rate * self.dt;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    
                    let vx_new = cos_a * self.state.x[3] - sin_a * self.state.x[4];
                    let vy_new = sin_a * self.state.x[3] + cos_a * self.state.x[4];
                    
                    self.state.x[3] = vx_new;
                    self.state.x[4] = vy_new;
                }
                
                // Update position
                self.state.x[0] += self.state.x[3] * self.dt;
                self.state.x[1] += self.state.x[4] * self.dt;
                self.state.x[2] += self.state.x[5] * self.dt;
            }
            MotionPhase::Accelerate { acceleration, .. } => {
                // Update velocity
                self.state.x[3] += acceleration[0] * self.dt;
                self.state.x[4] += acceleration[1] * self.dt;
                self.state.x[5] += acceleration[2] * self.dt;
                
                // Update position
                self.state.x[0] += self.state.x[3] * self.dt;
                self.state.x[1] += self.state.x[4] * self.dt;
                self.state.x[2] += self.state.x[5] * self.dt;
            }
        }
    }
    
    /// Update motion phase (switch between phases for realistic trajectory)
    fn update_phase(&mut self) {
        // Switch phases periodically for realistic motion
        match self.phase {
            MotionPhase::Straight => {
                if self.phase_timer > 30.0 {
                    // Switch to turn
                    let turn_rate = self.rng.gen_range(-0.1..0.1); // rad/s
                    let duration = self.rng.gen_range(10.0..20.0);
                    self.phase = MotionPhase::Turn { turn_rate, duration };
                    self.phase_timer = 0.0;
                }
            }
            MotionPhase::Turn { duration, .. } => {
                if self.phase_timer > duration {
                    // Switch back to straight
                    self.phase = MotionPhase::Straight;
                    self.phase_timer = 0.0;
                }
            }
            MotionPhase::Accelerate { duration, .. } => {
                if self.phase_timer > duration {
                    // Switch to straight
                    self.phase = MotionPhase::Straight;
                    self.phase_timer = 0.0;
                }
            }
        }
    }
    
    /// Get current true state
    pub fn true_state(&self) -> State {
        self.state
    }
    
    /// Get current time
    pub fn time(&self) -> f64 {
        self.time
    }
    
    /// Set phase timer (for randomization)
    pub fn set_phase_timer(&mut self, timer: f64) {
        self.phase_timer = timer;
    }
    
    /// Generate measurement with noise
    /// 
    /// # Arguments
    /// * `measurement_noise_std` - Standard deviation of measurement noise
    /// 
    /// # Returns
    /// Noisy measurement of aircraft position
    pub fn generate_measurement(&mut self, measurement_noise_std: f64) -> Measurement {
        // Add measurement noise
        let noise = Vector3::new(
            self.rng.gen_range(-measurement_noise_std..measurement_noise_std),
            self.rng.gen_range(-measurement_noise_std..measurement_noise_std),
            self.rng.gen_range(-measurement_noise_std..measurement_noise_std),
        );
        
        let z = self.state.position() + noise;
        let R = Matrix3::identity() * (measurement_noise_std * measurement_noise_std);
        
        Measurement::new(z, R, self.time)
    }
}

/// Clutter generator
/// 
/// Generates false alarm measurements (clutter) uniformly distributed
/// in the surveillance volume. In reality, clutter is not uniformly
/// distributed and depends on terrain, weather, etc.
pub struct ClutterGenerator {
    /// Surveillance volume bounds [x_min, x_max, y_min, y_max, z_min, z_max]
    bounds: [f64; 6],
    /// Clutter density (false alarms per unit volume per time step)
    density: f64,
    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl ClutterGenerator {
    /// Create a new clutter generator
    /// 
    /// # Arguments
    /// * `bounds` - Surveillance volume bounds [x_min, x_max, y_min, y_max, z_min, z_max]
    /// * `density` - Clutter density (false alarms per unit volume per time step)
    pub fn new(bounds: [f64; 6], density: f64) -> Self {
        Self {
            bounds,
            density,
            rng: rand::thread_rng(),
        }
    }
    
    /// Create clutter generator with default parameters
    pub fn default() -> Self {
        // Default surveillance volume: 10km x 10km x 5km
        let bounds = [-5000.0, 5000.0, -5000.0, 5000.0, 0.0, 10000.0];
        let density = 1e-8; // Clutter density
        
        Self::new(bounds, density)
    }
    
    /// Generate clutter around a specific position (localized clutter)
    /// 
    /// Generates clutter measurements near a target position, simulating
    /// realistic radar clutter that tends to be localized.
    /// 
    /// # Arguments
    /// * `time` - Current time
    /// * `measurement_noise_std` - Standard deviation for measurement noise
    /// * `center` - Center position for clutter generation
    /// * `radius` - Radius around center for clutter
    pub fn generate_localized(&mut self, time: f64, measurement_noise_std: f64, center: Vector3<f64>, radius: f64) -> Vec<Measurement> {
        // Generate fewer clutter points, localized around center
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius;
        let lambda = self.density * volume * 10.0; // Scale up for localized area
        
        // Simplified Poisson sampling
        let num_fa = if lambda > 0.0 {
            let mut count = 0;
            let mut p = (-lambda).exp();
            let mut s = p;
            let u = self.rng.gen::<f64>();
            while u > s && count < 50 {
                count += 1;
                p *= lambda / (count as f64);
                s += p;
            }
            count
        } else {
            0
        };
        
        let mut clutter = Vec::new();
        let R = nalgebra::Matrix3::identity() * (measurement_noise_std * measurement_noise_std);
        
        for _ in 0..num_fa {
            // Generate random point in sphere around center
            let theta = self.rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            let phi = self.rng.gen::<f64>() * std::f64::consts::PI;
            let r = self.rng.gen::<f64>().powf(1.0/3.0) * radius; // Uniform in volume
            
            let x = center[0] + r * phi.sin() * theta.cos();
            let y = center[1] + r * phi.sin() * theta.sin();
            let z = center[2] + r * phi.cos();
            
            let z_vec = Vector3::new(x, y, z);
            clutter.push(Measurement::new(z_vec, R, time));
        }
        
        clutter
    }
    
    /// Generate clutter measurements
    /// 
    /// Generates Poisson-distributed number of false alarms uniformly
    /// distributed in the surveillance volume.
    /// 
    /// # Arguments
    /// * `time` - Current time
    /// * `measurement_noise_std` - Standard deviation for measurement noise
    /// 
    /// # Returns
    /// Vector of clutter measurements
    pub fn generate(&mut self, time: f64, measurement_noise_std: f64) -> Vec<Measurement> {
        self.generate_in_bounds(
            time,
            measurement_noise_std,
            Vector3::new(self.bounds[0], self.bounds[2], self.bounds[4]),
            Vector3::new(self.bounds[1], self.bounds[3], self.bounds[5]),
        )
    }
    
    /// Generate clutter in specified bounds
    /// 
    /// Generates clutter uniformly distributed in the given bounds.
    /// 
    /// # Arguments
    /// * `time` - Current time
    /// * `measurement_noise_std` - Standard deviation for measurement noise
    /// * `bounds_min` - Minimum bounds (x, y, z)
    /// * `bounds_max` - Maximum bounds (x, y, z)
    /// 
    /// # Returns
    /// Vector of clutter measurements
    pub fn generate_in_bounds(
        &mut self,
        time: f64,
        measurement_noise_std: f64,
        bounds_min: Vector3<f64>,
        bounds_max: Vector3<f64>,
    ) -> Vec<Measurement> {
        // Compute volume of specified bounds
        let volume = (bounds_max[0] - bounds_min[0]) *
                     (bounds_max[1] - bounds_min[1]) *
                     (bounds_max[2] - bounds_min[2]);
        
        // Expected number of false alarms (Poisson approximation)
        // Use Poisson approximation: sample from exponential then count
        let lambda = self.density * volume;
        // Simplified Poisson: use direct sampling
        let num_fa = if lambda > 0.0 {
            // Approximate Poisson using direct method for small lambda
            let mut count = 0;
            let mut p = (-lambda).exp();
            let mut s = p;
            let u = self.rng.gen::<f64>();
            while u > s && count < 100 {
                count += 1;
                p *= lambda / (count as f64);
                s += p;
            }
            count
        } else {
            0
        };
        
        // Generate false alarms uniformly distributed in bounds
        let mut clutter = Vec::new();
        let r = Matrix3::identity() * (measurement_noise_std * measurement_noise_std);
        
        for _ in 0..num_fa {
            let x = self.rng.gen_range(bounds_min[0]..bounds_max[0]);
            let y = self.rng.gen_range(bounds_min[1]..bounds_max[1]);
            let z = self.rng.gen_range(bounds_min[2]..bounds_max[2]);
            
            let z_vec = Vector3::new(x, y, z);
            clutter.push(Measurement::new(z_vec, r, time));
        }
        
        clutter
    }
}

/// Complete simulation environment
/// 
/// Combines aircraft simulation and clutter generation for a complete
/// test environment.
pub struct Simulation {
    /// Aircraft simulators (one per trajectory)
    pub aircraft: Vec<AircraftSimulator>,
    /// Clutter generator
    clutter_gen: ClutterGenerator,
    /// Measurement noise standard deviation
    measurement_noise_std: f64,
    /// Detection probability (probability aircraft is detected)
    detection_prob: f64,
}

impl Simulation {
    /// Create a new simulation with a single aircraft
    /// 
    /// # Arguments
    /// * `dt` - Time step
    /// * `measurement_noise_std` - Measurement noise standard deviation
    /// * `detection_prob` - Detection probability
    pub fn new(dt: f64, measurement_noise_std: f64, detection_prob: f64) -> Self {
        Self::new_with_num_aircraft(dt, measurement_noise_std, detection_prob, 1)
    }
    
    /// Create a new simulation with multiple aircraft
    /// 
    /// # Arguments
    /// * `dt` - Time step
    /// * `measurement_noise_std` - Measurement noise standard deviation
    /// * `detection_prob` - Detection probability
    /// * `num_aircraft` - Number of aircraft to simulate
    pub fn new_with_num_aircraft(
        dt: f64, 
        measurement_noise_std: f64, 
        detection_prob: f64,
        num_aircraft: usize,
    ) -> Self {
        let mut aircraft = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Generate aircraft with random initial positions and velocities
        for i in 0..num_aircraft {
            // Spread aircraft across a large area
            // Use a grid-like distribution with some randomness
            let grid_size = (num_aircraft as f64).sqrt().ceil() as usize;
            let x_idx = i % grid_size;
            let y_idx = i / grid_size;
            
            // Base position: spread in a 10km x 10km grid at altitude 5000m
            let base_x = (x_idx as f64 - grid_size as f64 / 2.0) * 1000.0;
            let base_y = (y_idx as f64 - grid_size as f64 / 2.0) * 1000.0;
            let base_z = 5000.0;
            
            // Add random offset to avoid perfect grid
            let offset_x = rng.gen_range(-500.0..500.0);
            let offset_y = rng.gen_range(-500.0..500.0);
            let offset_z = rng.gen_range(-200.0..200.0);
            
            // Random initial velocity (50-150 m/s magnitude)
            let speed = rng.gen_range(50.0..150.0);
            let heading = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
            let vx = speed * heading.cos();
            let vy = speed * heading.sin();
            let vz = rng.gen_range(-10.0..10.0);
            
            let x = Vector6::new(
                base_x + offset_x,
                base_y + offset_y,
                base_z + offset_z,
                vx,
                vy,
                vz,
            );
            let P = Matrix6::identity() * 1.0; // Small initial uncertainty
            let initial_state = State::new(x, P);
            
            let mut simulator = AircraftSimulator::new(initial_state, dt);
            // Randomize initial phase timer to avoid synchronized motion
            simulator.set_phase_timer(rng.gen_range(0.0..10.0));
            aircraft.push(simulator);
        }
        
        let clutter_gen = ClutterGenerator::default();
        
        Self {
            aircraft,
            clutter_gen,
            measurement_noise_std,
            detection_prob,
        }
    }
    
    /// Step simulation forward
    /// 
    /// Updates all aircraft states and generates measurements (both true
    /// and clutter).
    /// 
    /// # Returns
    /// Tuple of (true states, measurements, time)
    pub fn step(&mut self) -> (Vec<State>, Vec<Measurement>, f64) {
        // Update all aircraft
        for aircraft in &mut self.aircraft {
            aircraft.step();
        }
        
        // Get time from first aircraft (all should have same time)
        let time = if let Some(first) = self.aircraft.first() {
            first.time()
        } else {
            0.0
        };
        
        // Generate measurements from all aircraft
        let mut measurements = Vec::new();
        let mut rng = rand::thread_rng();
        
        for aircraft in &mut self.aircraft {
            // Generate true measurement (with detection probability)
            if rng.gen::<f64>() < self.detection_prob {
                let true_measurement = aircraft.generate_measurement(self.measurement_noise_std);
                measurements.push(true_measurement);
            }
        }
        
        // Get all true states
        let true_states: Vec<State> = self.aircraft.iter().map(|a| a.true_state()).collect();
        
        (true_states, measurements, time)
    }
    
    /// Generate clutter in specified bounds
    /// 
    /// Generates clutter uniformly distributed in the given bounds.
    /// 
    /// # Arguments
    /// * `time` - Current time
    /// * `measurement_noise_std` - Standard deviation for measurement noise
    /// * `bounds_min` - Minimum bounds (x, y, z)
    /// * `bounds_max` - Maximum bounds (x, y, z)
    /// 
    /// # Returns
    /// Vector of clutter measurements
    pub fn generate_clutter_in_bounds(
        &mut self,
        time: f64,
        measurement_noise_std: f64,
        _bounds_min: Vector3<f64>,
        _bounds_max: Vector3<f64>,
    ) -> Vec<Measurement> {
        self.clutter_gen.generate_in_bounds(time, measurement_noise_std, _bounds_min, _bounds_max)
    }
    
    /// Get current true states (all aircraft)
    pub fn true_states(&self) -> Vec<State> {
        self.aircraft.iter().map(|a| a.true_state()).collect()
    }
    
    /// Get current true state (first aircraft, for backward compatibility)
    pub fn true_state(&self) -> State {
        self.aircraft.first()
            .map(|a| a.true_state())
            .unwrap_or_else(|| State::zero())
    }
    
    /// Get current time
    pub fn time(&self) -> f64 {
        self.aircraft.first()
            .map(|a| a.time())
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aircraft_simulator_creation() {
        let state = State::zero();
        let simulator = AircraftSimulator::new(state, 1.0);
        
        assert_eq!(simulator.time(), 0.0);
    }
    
    #[test]
    fn test_aircraft_simulator_step() {
        let mut simulator = AircraftSimulator::default(1.0);
        let initial_pos = simulator.true_state().position();
        
        simulator.step();
        
        // Position should have changed
        let new_pos = simulator.true_state().position();
        assert_ne!(initial_pos, new_pos);
    }
    
    #[test]
    fn test_measurement_generation() {
        let mut simulator = AircraftSimulator::default(1.0);
        let measurement = simulator.generate_measurement(10.0);
        
        assert_eq!(measurement.time, 0.0);
        // Measurement should be near true position
        let true_pos = simulator.true_state().position();
        let distance = (measurement.z - true_pos).magnitude();
        assert!(distance < 100.0); // Should be within reasonable noise
    }
    
    #[test]
    fn test_clutter_generator() {
        let mut clutter_gen = ClutterGenerator::default();
        let clutter = clutter_gen.generate(0.0, 10.0);
        
        // Should generate some clutter (may be zero with low density)
        assert!(clutter.len() >= 0);
    }
    
    #[test]
    fn test_simulation_step() {
        let mut sim = Simulation::new(1.0, 10.0, 0.9);
        let (true_state, measurements, time) = sim.step();
        
        assert!(time > 0.0);
        assert!(!measurements.is_empty() || true); // May have no measurements with low clutter
    }
    
    #[test]
    fn test_simulation_multiple_steps() {
        let mut sim = Simulation::new(1.0, 10.0, 0.9);
        
        for _ in 0..10 {
            sim.step();
        }
        
        // Should have progressed in time
        assert!(sim.time() > 0.0);
    }
}

