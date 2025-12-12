//! 3D visualization using egui
//! 
//! This module provides a 3D visualization of the tracking system,
//! showing:
//! - True aircraft trajectory
//! - Track estimates
//! - Measurements (true + clutter)
//! - Real-time tracking performance

use eframe::egui;
use nalgebra::Vector3;
use rand::Rng;
use crate::state::{State, Measurement, Track};
use crate::simulation::Simulation;
use crate::tracker::Tracker;

/// Main application for visualization
pub struct TrackerApp {
    /// Simulation
    simulation: Simulation,
    /// Tracker
    tracker: Tracker,
    /// History of true states
    true_history: Vec<State>,
    /// History of track estimates
    track_history: Vec<State>,
    /// History of measurements
    measurement_history: Vec<Vec<Measurement>>,
    /// Simulation running flag
    running: bool,
    /// Simulation speed (steps per frame)
    speed: usize,
    /// View parameters
    view_center: Vector3<f64>,
    view_scale: f64,
    /// View offset for scrolling
    view_offset: egui::Vec2,
    /// Auto-follow aircraft flag
    auto_follow: bool,
    /// Show clutter flag
    show_clutter: bool,
    /// Show true trajectory flag
    show_true: bool,
    /// Show track estimate flag
    show_track: bool,
    /// Show tracking error lines
    show_error_lines: bool,
    /// Auto-stop after this many steps (None = no auto-stop)
    auto_stop_after_steps: Option<usize>,
    /// Step counter for auto-stop
    step_counter: usize,
}

impl Default for TrackerApp {
    fn default() -> Self {
        let dt = 1.0; // 1 second time step
        let simulation = Simulation::new(dt, 10.0, 0.9);
        let tracker = Tracker::new(dt);
        
        Self {
            simulation,
            tracker,
            true_history: Vec::new(),
            track_history: Vec::new(),
            measurement_history: Vec::new(),
            running: true, // Auto-play at startup
            speed: 1,
            auto_stop_after_steps: Some(10000), // Auto-stop after 100 steps for analysis
            view_center: Vector3::new(0.0, 0.0, 5000.0),
            view_scale: 0.1, // Smaller scale for better visibility
            view_offset: egui::Vec2::ZERO,
            auto_follow: true, // Auto-follow aircraft by default
            show_clutter: true,
            show_true: true,
            show_track: true,
            show_error_lines: true,
            step_counter: 0,
        }
    }
}

impl TrackerApp {
    /// Get view bounds in world coordinates
    /// 
    /// Computes the bounds of the visible area in world coordinates
    /// based on view center, scale, and offset.
    fn get_view_bounds(&self, rect: egui::Rect) -> (Vector3<f64>, Vector3<f64>) {
        // Calculate visible area in world coordinates
        let half_width = (rect.width() / 2.0 / self.view_scale as f32) as f64;
        let half_height = (rect.height() / 2.0 / self.view_scale as f32) as f64;
        
        // Account for view offset
        let offset_x = self.view_offset.x as f64 / self.view_scale;
        let offset_y = self.view_offset.y as f64 / self.view_scale;
        
        let min = Vector3::new(
            self.view_center[0] - half_width + offset_x,
            self.view_center[1] - half_height - offset_y, // Note: y is inverted
            self.view_center[2] - 1000.0, // Z range
        );
        
        let max = Vector3::new(
            self.view_center[0] + half_width + offset_x,
            self.view_center[1] + half_height - offset_y,
            self.view_center[2] + 1000.0,
        );
        
        (min, max)
    }
    /// Step simulation and tracker
    fn step(&mut self) {
        for _ in 0..self.speed {
            self.step_counter += 1;
            
            // Auto-stop if configured
            if let Some(max_steps) = self.auto_stop_after_steps {
                if self.step_counter >= max_steps {
                    self.running = false;
                    log::info!("Auto-stopped after {} steps", max_steps);
                    return;
                }
            }
            // Step simulation
            let (true_state, measurements, time) = self.simulation.step();
            
            // Generate clutter in view bounds (view-adaptive)
            // This ensures clutter density is constant regardless of zoom level
            // Note: We'll generate clutter based on current view when drawing
            // For now, generate minimal clutter to avoid performance issues
            
            // Update tracker
            self.tracker.update(&measurements, time);
            
            // Store history
            self.true_history.push(true_state);
            self.measurement_history.push(measurements);
            
            // Store track estimate
            if let Some(track) = self.tracker.primary_track() {
                self.track_history.push(track.state);
            }
            
            // Auto-follow aircraft: update view center to follow current position
            if self.auto_follow {
                if let Some(last_state) = self.true_history.last() {
                    // Smoothly follow the aircraft (use aircraft position as view center)
                    self.view_center = last_state.position();
                    // Keep Z at a reasonable viewing height
                    self.view_center[2] = 5000.0;
                }
            }
            
            // Limit history size
            if self.true_history.len() > 1000 {
                self.true_history.remove(0);
                self.track_history.remove(0);
                self.measurement_history.remove(0);
            }
        }
    }
    
    /// Draw 3D scene
    fn draw_3d(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        // Get painter
        let painter = ui.painter();
        
        // Transform coordinates to screen space
        // Simplified 2D projection (top-down view with height as color)
        let to_screen = |pos: Vector3<f64>| -> egui::Pos2 {
            let offset = pos - self.view_center;
            let offset_x = if self.auto_follow { 0.0 } else { self.view_offset.x };
            let offset_y = if self.auto_follow { 0.0 } else { self.view_offset.y };
            let x = rect.center().x + (offset[0] * self.view_scale) as f32 - offset_x;
            let y = rect.center().y - (offset[1] * self.view_scale) as f32 - offset_y;
            egui::Pos2::new(x, y)
        };
        
        // Draw measurements (clutter)
        if self.show_clutter {
            for measurements in &self.measurement_history {
                for measurement in measurements {
                    let screen_pos = to_screen(measurement.z);
                    painter.circle_filled(screen_pos, 2.0, egui::Color32::from_rgb(100, 100, 100));
                }
            }
        }
        
        // Draw true trajectory
        if self.show_true && self.true_history.len() > 1 {
            let mut points = Vec::new();
            for state in &self.true_history {
                points.push(to_screen(state.position()));
            }
            
            if points.len() > 1 {
                painter.add(egui::Shape::line(
                    points,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 255, 0)),
                ));
            }
            
            // Draw current true position
            if let Some(last_state) = self.true_history.last() {
                let screen_pos = to_screen(last_state.position());
                painter.circle_filled(screen_pos, 5.0, egui::Color32::from_rgb(0, 255, 0));
            }
        }
        
        // Draw track estimate
        if self.show_track && self.track_history.len() > 1 {
            let mut points = Vec::new();
            for state in &self.track_history {
                points.push(to_screen(state.position()));
            }
            
            if points.len() > 1 {
                painter.add(egui::Shape::line(
                    points,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 0, 0)),
                ));
            }
            
            // Draw current track estimate
            if let Some(last_state) = self.track_history.last() {
                let screen_pos = to_screen(last_state.position());
                painter.circle_filled(screen_pos, 5.0, egui::Color32::from_rgb(255, 0, 0));
                
                // Draw uncertainty ellipse (simplified as circle)
                let uncertainty = last_state.P[(0, 0)].sqrt().max(last_state.P[(1, 1)].sqrt());
                let radius = (uncertainty * self.view_scale) as f32;
                painter.circle_stroke(screen_pos, radius, egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 100, 100)));
            }
        }
        
        // Draw tracking error lines (connect true position to track estimate)
        if self.show_error_lines && self.show_true && self.show_track {
            if let (Some(true_state), Some(track_state)) = (self.true_history.last(), self.track_history.last()) {
                let true_pos = to_screen(true_state.position());
                let track_pos = to_screen(track_state.position());
                
                // Draw line connecting true and estimated positions
                painter.line_segment(
                    [true_pos, track_pos],
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 255, 0, 150)),
                );
                
                // Compute and display tracking error
                let error = (true_state.position() - track_state.position()).magnitude();
                let mid_point = egui::Pos2::new(
                    (true_pos.x + track_pos.x) / 2.0,
                    (true_pos.y + track_pos.y) / 2.0,
                );
                
                // Draw error text (small)
                painter.text(
                    mid_point,
                    egui::Align2::CENTER_CENTER,
                    format!("{:.0}m", error),
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(255, 255, 0),
                );
            }
        }
    }
}

impl eframe::App for TrackerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Step simulation if running
        if self.running {
            self.step();
            ctx.request_repaint();
        }
        
        // Right panel with tracks list
        egui::SidePanel::right("tracks_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Live Tracks");
                ui.separator();
                
                let tracks = self.tracker.tracks();
                if tracks.is_empty() {
                    ui.label("No active tracks");
                } else {
                    ui.label(format!("Active Tracks: {}", tracks.len()));
                    ui.separator();
                    
                    for track in tracks {
                        ui.group(|ui| {
                            ui.label(format!("Track ID: {}", track.id));
                            ui.label(format!("Age: {} steps", track.age));
                            ui.label(format!("Missed: {}", track.missed_detections));
                            ui.label(format!("Existence: {:.2}", track.existence_prob));
                            
                            let pos = track.state.position();
                            ui.label(format!("Position: ({:.0}, {:.0}, {:.0})", pos[0], pos[1], pos[2]));
                            
                            let vel = track.state.velocity();
                            let speed = vel.magnitude();
                            ui.label(format!("Speed: {:.1} m/s", speed));
                            
                            // Model probabilities
                            ui.separator();
                            ui.label("Model Probabilities:");
                            if track.model_probs.len() >= 3 {
                                ui.label(format!("  CV: {:.2}%", track.model_probs[0] * 100.0));
                                ui.label(format!("  CA: {:.2}%", track.model_probs[1] * 100.0));
                                ui.label(format!("  CT: {:.2}%", track.model_probs[2] * 100.0));
                            }
                            
                            // Tracking error if true state available
                            if let Some(true_state) = self.true_history.last() {
                                let error = (true_state.position() - track.state.position()).magnitude();
                                let error_color = if error < 50.0 {
                                    egui::Color32::from_rgb(0, 255, 0) // Green for good
                                } else if error < 200.0 {
                                    egui::Color32::from_rgb(255, 255, 0) // Yellow for moderate
                                } else {
                                    egui::Color32::from_rgb(255, 0, 0) // Red for poor
                                };
                                ui.label(egui::RichText::new(format!("Error: {:.1} m", error)).color(error_color));
                            }
                        });
                        ui.add_space(5.0);
                    }
                }
            });
        
        // Main UI
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("AI Tracker - Multi-Target Tracking");
            
            // Control panel
            ui.horizontal(|ui| {
                if ui.button(if self.running { "Pause" } else { "Play" }).clicked() {
                    self.running = !self.running;
                }
                
                if ui.button("Step").clicked() {
                    self.step();
                }
                
                if ui.button("Reset").clicked() {
                    *self = Self::default();
                }
                
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed, 1..=10));
            });
            
            ui.separator();
            
            // Display info
            ui.horizontal(|ui| {
                ui.label(format!("Time: {:.1} s", self.simulation.time()));
                
                if let Some(track) = self.tracker.primary_track() {
                    ui.label(format!("Track Age: {}", track.age));
                    ui.label(format!("Missed: {}", track.missed_detections));
                    ui.label(format!("Existence: {:.2}", track.existence_prob));
                } else {
                    ui.label("No track");
                }
            });
            
            ui.separator();
            
            // View controls
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_true, "Show True Trajectory");
                ui.checkbox(&mut self.show_track, "Show Track Estimate");
                ui.checkbox(&mut self.show_clutter, "Show Clutter");
                ui.checkbox(&mut self.show_error_lines, "Show Error Lines");
            });
            
            ui.horizontal(|ui| {
                ui.label("View Scale:");
                ui.add(egui::Slider::new(&mut self.view_scale, 0.01..=10.0));
                ui.checkbox(&mut self.auto_follow, "Auto-Follow");
            });
            
            ui.separator();
            
            // 3D visualization area - resize with window
            let available_rect = ui.available_rect_before_wrap();
            // Use all available space, accounting for control panel height
            let control_panel_height = 200.0; // Approximate height of controls
            let plot_rect = egui::Rect::from_min_size(
                egui::Pos2::new(available_rect.min.x, available_rect.min.y + control_panel_height),
                egui::Vec2::new(
                    available_rect.width(),
                    (available_rect.height() - control_panel_height).max(100.0),
                ),
            );
            
            // Draw visualization area (resize with window)
            // Use ScrollArea only if not auto-following (for manual panning)
            if self.auto_follow {
                // Auto-follow mode: no scrolling, view follows aircraft
                self.view_offset = egui::Vec2::ZERO;
                
                // Generate clutter in current view bounds
                let view_bounds = self.get_view_bounds(plot_rect);
                let clutter = self.simulation.generate_clutter_in_bounds(
                    self.simulation.time(),
                    10.0,
                    view_bounds.0,
                    view_bounds.1,
                );
                
                // Draw background
                ui.painter().rect_filled(plot_rect, 0.0, egui::Color32::from_rgb(20, 20, 30));
                
                // Draw clutter in view
                if self.show_clutter {
                    for measurement in &clutter {
                        let screen_pos = {
                            let offset = measurement.z - self.view_center;
                            let x = plot_rect.center().x + (offset[0] * self.view_scale) as f32;
                            let y = plot_rect.center().y - (offset[1] * self.view_scale) as f32;
                            egui::Pos2::new(x, y)
                        };
                        ui.painter().circle_filled(screen_pos, 2.0, egui::Color32::from_rgb(100, 100, 100));
                    }
                }
                
                // Draw 3D scene
                self.draw_3d(ui, plot_rect);
            } else {
                // Manual panning mode: use ScrollArea
                egui::ScrollArea::both()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        // Update view offset based on scroll
                        let scroll_state = ui.ctx().data(|d| {
                            d.get_temp::<egui::Vec2>(egui::Id::new("tracker_scroll"))
                                .unwrap_or_default()
                        });
                        self.view_offset = scroll_state;
                        
                        // Generate clutter in current view bounds
                        let view_bounds = self.get_view_bounds(plot_rect);
                        let clutter = self.simulation.generate_clutter_in_bounds(
                            self.simulation.time(),
                            10.0,
                            view_bounds.0,
                            view_bounds.1,
                        );
                        
                        // Draw background
                        ui.painter().rect_filled(plot_rect, 0.0, egui::Color32::from_rgb(20, 20, 30));
                        
                        // Draw clutter in view
                        if self.show_clutter {
                            for measurement in &clutter {
                                let screen_pos = {
                                    let offset = measurement.z - self.view_center;
                                    let x = plot_rect.center().x + (offset[0] * self.view_scale) as f32 - self.view_offset.x;
                                    let y = plot_rect.center().y - (offset[1] * self.view_scale) as f32 - self.view_offset.y;
                                    egui::Pos2::new(x, y)
                                };
                                ui.painter().circle_filled(screen_pos, 2.0, egui::Color32::from_rgb(100, 100, 100));
                            }
                        }
                        
                        // Draw 3D scene
                        self.draw_3d(ui, plot_rect);
                        
                        // Store scroll state
                        ui.ctx().data_mut(|d| {
                            d.insert_temp(egui::Id::new("tracker_scroll"), self.view_offset);
                        });
                    });
            }
            
            // Legend
            let legend_rect = egui::Rect::from_min_size(
                egui::Pos2::new(plot_rect.min.x + 10.0, plot_rect.min.y + 10.0),
                egui::Vec2::new(200.0, 100.0),
            );
            
            egui::Frame::popup(ui.style())
                .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200))
                .show(ui, |ui| {
                    ui.set_clip_rect(legend_rect);
                    ui.label("Legend:");
                    ui.horizontal(|ui| {
                        let pos = ui.cursor().min;
                        ui.painter().circle_filled(egui::Pos2::new(pos.x + 5.0, pos.y + 5.0), 5.0, egui::Color32::from_rgb(0, 255, 0));
                        ui.label("True Position");
                    });
                    ui.horizontal(|ui| {
                        let pos = ui.cursor().min;
                        ui.painter().circle_filled(egui::Pos2::new(pos.x + 5.0, pos.y + 5.0), 5.0, egui::Color32::from_rgb(255, 0, 0));
                        ui.label("Track Estimate");
                    });
                    ui.horizontal(|ui| {
                        let pos = ui.cursor().min;
                        ui.painter().circle_filled(egui::Pos2::new(pos.x + 5.0, pos.y + 5.0), 2.0, egui::Color32::from_rgb(100, 100, 100));
                        ui.label("Clutter");
                    });
                    ui.horizontal(|ui| {
                        let pos = ui.cursor().min;
                        ui.painter().line_segment(
                            [egui::Pos2::new(pos.x, pos.y + 5.0), egui::Pos2::new(pos.x + 20.0, pos.y + 5.0)],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 255, 0)),
                        );
                        ui.label("Tracking Error");
                    });
                });
        });
    }
}

