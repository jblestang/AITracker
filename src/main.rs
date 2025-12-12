//! Main entry point for the AI Tracker application
//! 
//! This application provides a 3D visualization of multi-target tracking
//! using IMM and JPDA algorithms for aircraft tracking in clutter.

mod state;
mod kalman;
mod imm;
mod jpda;
mod simulation;
mod tracker;
mod visualization;

use eframe::egui;
use log::info;

/// Main application entry point
fn main() -> Result<(), eframe::Error> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    info!("Starting AI Tracker application");
    
    // Create the application options
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("AI Tracker - Multi-Target Tracking"),
        ..Default::default()
    };
    
    // Run the application
    eframe::run_native(
        "AI Tracker",
        options,
        Box::new(|_cc| Box::new(visualization::TrackerApp::default())),
    )
}

