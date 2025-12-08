//! String Theory Landscape Explorer
//!
//! A genetic algorithm searching for the compactification parameters
//! that reproduce our universe's physical constants.
//!
//! Now with REAL physics via JAX/cymyc!

mod compactification;
mod constants;
mod fitness;
mod genetic;
mod physics;
mod real_genetic;
mod renderer;

use std::sync::Arc;
use std::time::{Duration, Instant};

use fitness::{format_fitness_report, format_fitness_line};
use genetic::{GaConfig, LandscapeSearcher};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Auto-save interval in seconds
const AUTO_SAVE_INTERVAL: u64 = 30;

/// Application state
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<renderer::Renderer>,
    searcher: LandscapeSearcher,
    last_report: Instant,
    last_save: Instant,
    generations_per_frame: usize,
    paused: bool,
    last_reported_fitness: f64,
}

impl App {
    fn new() -> Self {
        // Configure the GA with advanced techniques
        let config = GaConfig {
            population_size: 2000,
            elite_count: 20,
            tournament_size: 5,
            crossover_rate: 0.85,
            base_mutation_rate: 0.4,
            base_mutation_strength: 0.35,
            collapse_threshold: 5000,  // Landscape collapse after 5000 stagnant generations
            hall_of_fame_size: 100,  // Keep 100 best solutions in archive
        };

        println!("═══════════════════════════════════════════════════════════════");
        println!("  STRING THEORY LANDSCAPE EXPLORER");
        println!("  Searching for our universe's compactification...");
        println!("═══════════════════════════════════════════════════════════════");
        println!();
        println!("  Parameter space dimension: {}", compactification::Compactification::parameter_count());
        println!("  Population size: {}", config.population_size);
        println!("  Target constants:");
        println!("    α_em = {:.6e} (fine structure)", constants::ALPHA_EM);
        println!("    α_s  = {:.4} (strong coupling)", constants::ALPHA_STRONG);
        println!("    sin²θ_W = {:.5} (Weinberg angle)", constants::SIN2_THETA_W);
        println!("    m_e/M_Pl = {:.3e}", constants::ELECTRON_PLANCK_RATIO);
        println!("    m_p/M_Pl = {:.3e}", constants::PROTON_PLANCK_RATIO);
        println!("    Λ = {:.3e} (cosmological constant)", constants::COSMOLOGICAL_CONSTANT);
        println!("    N_gen = {} (fermion generations)", constants::NUM_GENERATIONS);
        println!();
        println!("  Controls:");
        println!("    SPACE - pause/resume");
        println!("    +/-   - adjust speed");
        println!("    R     - print detailed report");
        println!("    S     - save state");
        println!("    Q/ESC - quit (auto-saves)");
        println!();

        let searcher = LandscapeSearcher::load_or_new(config);

        // Print initial best
        if let Some(best) = searcher.best() {
            println!("Initial best:");
            println!("{}", format_fitness_report(best));
        }

        Self {
            window: None,
            renderer: None,
            searcher,
            last_report: Instant::now(),
            last_save: Instant::now(),
            generations_per_frame: 5,
            paused: false,
            last_reported_fitness: 0.0,
        }
    }

    fn save_state(&mut self) {
        match self.searcher.save_state() {
            Ok(()) => println!("💾 State saved to string_theory_state.json"),
            Err(e) => eprintln!("Failed to save state: {}", e),
        }
        self.last_save = Instant::now();
    }

    fn run_generations(&mut self) {
        if self.paused {
            return;
        }

        for _ in 0..self.generations_per_frame {
            self.searcher.step();
        }

        // Console output only when fitness improves
        if let Some(best) = self.searcher.best() {
            if best.fitness > self.last_reported_fitness * 1.0001 {
                // New best found!
                self.last_reported_fitness = best.fitness;
                let gen = self.searcher.generation;
                let total = self.searcher.total_evaluated;
                let collapses = self.searcher.history.last()
                    .map(|s| s.landscape_collapses)
                    .unwrap_or(0);
                println!(
                    "🎯 Gen {:6} | Eval: {:>10} | 🌠:{} | {}",
                    gen, total, collapses, format_fitness_line(best)
                );
            } else if self.last_report.elapsed() > Duration::from_secs(10) {
                // Periodic status update (less frequent when stagnant)
                self.last_report = Instant::now();
                let gen = self.searcher.generation;
                let total = self.searcher.total_evaluated;
                let stag = self.searcher.history.last()
                    .map(|s| s.stagnation_generations)
                    .unwrap_or(0);
                let collapses = self.searcher.history.last()
                    .map(|s| s.landscape_collapses)
                    .unwrap_or(0);
                println!(
                    "   Gen {:6} | Eval: {:>10} | stag:{:4} | 🌠:{}",
                    gen, total, stag, collapses
                );
            }
        }

        // Auto-save periodically
        if self.last_save.elapsed() > Duration::from_secs(AUTO_SAVE_INTERVAL) {
            self.save_state();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("String Theory Landscape Explorer")
                .with_inner_size(LogicalSize::new(1200, 800));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
            self.window = Some(window.clone());

            // Initialize renderer
            let renderer = pollster::block_on(renderer::Renderer::new(window));
            self.renderer = Some(renderer);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Save state before exiting
                self.save_state();

                println!();
                println!("═══════════════════════════════════════════════════════════════");
                println!("  FINAL REPORT");
                println!("═══════════════════════════════════════════════════════════════");
                println!("  Generations: {}", self.searcher.generation);
                println!("  Total evaluated: {}", self.searcher.total_evaluated);
                println!();

                if let Some(best) = self.searcher.best_ever.as_ref() {
                    println!("Best ever found:");
                    println!("{}", format_fitness_report(best));
                }

                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    match event.logical_key.as_ref() {
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                            self.paused = !self.paused;
                            println!(
                                "{}",
                                if self.paused {
                                    "PAUSED"
                                } else {
                                    "RESUMED"
                                }
                            );
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                            self.save_state();
                            event_loop.exit();
                        }
                        winit::keyboard::Key::Character(c) => match c.to_string().as_str() {
                            "q" | "Q" => {
                                self.save_state();
                                event_loop.exit();
                            }
                            "r" | "R" => {
                                if let Some(best) = self.searcher.best() {
                                    println!();
                                    println!("{}", format_fitness_report(best));
                                }
                            }
                            "s" | "S" => {
                                self.save_state();
                            }
                            "+" | "=" => {
                                self.generations_per_frame =
                                    (self.generations_per_frame + 1).min(50);
                                println!(
                                    "Speed: {} generations/frame",
                                    self.generations_per_frame
                                );
                            }
                            "-" | "_" => {
                                self.generations_per_frame =
                                    (self.generations_per_frame - 1).max(1);
                                println!(
                                    "Speed: {} generations/frame",
                                    self.generations_per_frame
                                );
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // Run GA iterations
                self.run_generations();

                // Update and render
                if let Some(renderer) = &mut self.renderer {
                    renderer.update(
                        &self.searcher.population,
                        &self.searcher.history,
                        self.searcher.best_ever.as_ref(),
                        self.searcher.generation,
                        self.searcher.total_evaluated,
                    );

                    match renderer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            if let Some(window) = &self.window {
                                renderer.resize(window.inner_size());
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            event_loop.exit();
                        }
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }

                // Request another frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
