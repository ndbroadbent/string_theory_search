//! String Theory Landscape Explorer
//!
//! A genetic algorithm searching for the compactification parameters
//! that reproduce our universe's physical constants.
//!
//! "Like mining bitcoin on a 486 laptop" - but we might learn something.

mod compactification;
mod constants;
mod fitness;
mod genetic;
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

/// Application state
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<renderer::Renderer>,
    searcher: LandscapeSearcher,
    last_update: Instant,
    last_report: Instant,
    generations_per_frame: usize,
    paused: bool,
}

impl App {
    fn new() -> Self {
        // Configure the GA with advanced techniques
        let config = GaConfig {
            population_size: 2000,
            elite_count: 20,
            tournament_size: 5,
            crossover_rate: 0.85,
            base_mutation_rate: 0.15,
            base_mutation_strength: 0.15,
            asteroid_threshold: 50,  // Asteroid impact after 50 stagnant generations
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
        println!("    Q/ESC - quit");
        println!();

        let searcher = LandscapeSearcher::new(config);

        // Print initial best
        if let Some(best) = searcher.best() {
            println!("Initial best:");
            println!("{}", format_fitness_report(best));
        }

        Self {
            window: None,
            renderer: None,
            searcher,
            last_update: Instant::now(),
            last_report: Instant::now(),
            generations_per_frame: 5,
            paused: false,
        }
    }

    fn run_generations(&mut self) {
        if self.paused {
            return;
        }

        for _ in 0..self.generations_per_frame {
            self.searcher.step();
        }

        // Periodic console output
        if self.last_report.elapsed() > Duration::from_secs(2) {
            self.last_report = Instant::now();
            if let Some(best) = self.searcher.best() {
                let gen = self.searcher.generation;
                let total = self.searcher.total_evaluated;
                let stag = self.searcher.history.last()
                    .map(|s| s.stagnation_generations)
                    .unwrap_or(0);
                let impacts = self.searcher.history.last()
                    .map(|s| s.asteroid_impacts)
                    .unwrap_or(0);
                println!(
                    "Gen {:6} | Eval: {:>10} | stag:{:3} | 🌠:{} | {}",
                    gen, total, stag, impacts, format_fitness_line(best)
                );
            }
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
                            event_loop.exit();
                        }
                        winit::keyboard::Key::Character(c) => match c.to_string().as_str() {
                            "q" | "Q" => event_loop.exit(),
                            "r" | "R" => {
                                if let Some(best) = self.searcher.best() {
                                    println!();
                                    println!("{}", format_fitness_report(best));
                                }
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
                    renderer.update(&self.searcher.population, &self.searcher.history);

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
