//! Professional-grade GPU visualization of the string landscape search.
//!
//! Features:
//! - Population scatter plot with fitness-based coloring
//! - Real-time fitness history graph
//! - Live statistics panel with text rendering
//! - Physical constants comparison display

use bytemuck::{Pod, Zeroable};
use glyphon::{
    Attrs, Buffer, Cache, Color as GlyphonColor, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use std::sync::Arc;
use wgpu::MultisampleState;
use winit::window::Window;

use crate::constants::{TARGETS, TARGET_NAMES};
use crate::fitness::Individual;
use crate::genetic::GenerationStats;

/// Vertex for rendering shapes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Color palette for the UI
struct Colors;
impl Colors {
    const BACKGROUND: [f32; 4] = [0.0, 0.0, 0.0, 1.0];       // Pure black
    const PANEL_BG: [f32; 4] = [0.005, 0.005, 0.008, 1.0];   // Essentially black with hint of blue
    const PANEL_BORDER: [f32; 4] = [0.08, 0.15, 0.3, 0.4];
    const GRID_LINE: [f32; 4] = [0.06, 0.06, 0.08, 0.25];
    const ACCENT: [f32; 4] = [0.3, 0.7, 1.0, 1.0];
    const SUCCESS: [f32; 4] = [0.2, 0.9, 0.4, 1.0];
    const WARNING: [f32; 4] = [1.0, 0.8, 0.2, 1.0];
    const ERROR: [f32; 4] = [1.0, 0.3, 0.3, 1.0];
}

/// Smart number formatting: decimal for reasonable values, scientific for extreme
fn format_smart(v: f64) -> String {
    let abs_v = v.abs();
    if abs_v == 0.0 {
        "0".to_string()
    } else if abs_v >= 0.001 && abs_v < 10000.0 {
        // Show as decimal with appropriate precision
        if abs_v >= 1.0 {
            format!("{:.4}", v)
        } else if abs_v >= 0.01 {
            format!("{:.5}", v)
        } else {
            format!("{:.6}", v)
        }
    } else {
        // Scientific notation for very large or very small
        format!("{:.2e}", v)
    }
}

/// State for the renderer
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    max_vertices: usize,
    // Text rendering
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    text_buffers: Vec<(Buffer, f32, f32)>,  // (buffer, x, y)
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Shape shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let max_vertices = 200_000;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Text rendering setup
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            MultisampleState::default(),
            None,
        );

        let viewport = Viewport::new(&device, &cache);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            num_vertices: 0,
            max_vertices,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            text_buffers: Vec::new(),
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Update visualization with current state
    pub fn update(
        &mut self,
        population: &[Individual],
        history: &[GenerationStats],
        best_ever: Option<&Individual>,
        generation: usize,
        total_evaluated: u64,
    ) {
        let w = self.size.width as f32;
        let h = self.size.height as f32;

        let mut vertices = Vec::with_capacity(self.max_vertices);

        // Layout constants (in pixels, will convert to NDC)
        let margin = 20.0;
        let panel_padding = 20.0;

        // === LEFT PANEL: Statistics ===
        let stats_panel_w = 600.0;
        let stats_panel_h = h - 2.0 * margin;
        draw_panel(
            &mut vertices,
            margin,
            margin,
            stats_panel_w,
            stats_panel_h,
            w,
            h,
        );

        // === CENTER TOP: Population Scatter Plot ===
        let scatter_x = margin + stats_panel_w + margin;
        let scatter_w = w - scatter_x - margin;
        let scatter_h = (h - 3.0 * margin) * 0.55;
        draw_panel(&mut vertices, scatter_x, margin, scatter_w, scatter_h, w, h);

        // Grid lines for scatter plot
        let grid_x = scatter_x + panel_padding;
        let grid_y = margin + panel_padding + 30.0; // Leave room for title
        let grid_w = scatter_w - 2.0 * panel_padding;
        let grid_h = scatter_h - 2.0 * panel_padding - 40.0;

        draw_grid(&mut vertices, grid_x, grid_y, grid_w, grid_h, 10, 8, w, h);

        // Plot population points
        for ind in population.iter().take(2000) {
            // X: string coupling (0.01 to 0.5)
            let norm_x = ((ind.genome.string_coupling - 0.01) / 0.49).clamp(0.0, 1.0) as f32;
            // Y: log(CY volume) (0 to 7)
            let norm_y = (ind.genome.cy_volume.ln() / 7.0).clamp(0.0, 1.0) as f32;

            let px = grid_x + norm_x * grid_w;
            let py = grid_y + grid_h - norm_y * grid_h; // Flip Y

            let color = fitness_to_color(ind.fitness as f32);
            let size = 2.0 + (ind.fitness as f32).sqrt() * 8.0;

            draw_point(&mut vertices, px, py, size, color, w, h);
        }

        // === CENTER BOTTOM: Fitness History Graph ===
        let graph_y = margin + scatter_h + margin;
        let graph_h = h - graph_y - margin;
        draw_panel(
            &mut vertices,
            scatter_x,
            graph_y,
            scatter_w,
            graph_h,
            w,
            h,
        );

        // Graph area
        let gx = scatter_x + panel_padding;
        let gy = graph_y + panel_padding + 25.0;
        let gw = scatter_w - 2.0 * panel_padding;
        let gh = graph_h - 2.0 * panel_padding - 35.0;

        draw_grid(&mut vertices, gx, gy, gw, gh, 10, 5, w, h);

        // Plot fitness history
        if history.len() > 1 {
            let max_points = 1000;
            let step = (history.len() / max_points).max(1);
            let points: Vec<_> = history.iter().step_by(step).collect();

            // Best fitness line (green)
            for i in 0..points.len().saturating_sub(1) {
                let x1 = gx + (i as f32 / points.len() as f32) * gw;
                let x2 = gx + ((i + 1) as f32 / points.len() as f32) * gw;

                let y1 = gy + gh
                    - ((points[i].best_fitness.log10().max(-15.0) + 15.0) / 15.0) as f32 * gh;
                let y2 = gy + gh
                    - ((points[i + 1].best_fitness.log10().max(-15.0) + 15.0) / 15.0) as f32 * gh;

                draw_line(&mut vertices, x1, y1, x2, y2, 2.0, Colors::SUCCESS, w, h);
            }

            // Average fitness line (blue, dimmer)
            for i in 0..points.len().saturating_sub(1) {
                let x1 = gx + (i as f32 / points.len() as f32) * gw;
                let x2 = gx + ((i + 1) as f32 / points.len() as f32) * gw;

                let y1 = gy + gh
                    - ((points[i].avg_fitness.log10().max(-15.0) + 15.0) / 15.0) as f32 * gh;
                let y2 = gy + gh
                    - ((points[i + 1].avg_fitness.log10().max(-15.0) + 15.0) / 15.0) as f32 * gh;

                draw_line(&mut vertices, x1, y1, x2, y2, 1.5, [0.3, 0.5, 0.8, 0.7], w, h);
            }
        }

        // Truncate vertices
        if vertices.len() > self.max_vertices {
            vertices.truncate(self.max_vertices);
        }
        self.num_vertices = vertices.len() as u32;

        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));

        // === TEXT RENDERING ===
        self.text_buffers.clear();

        // Title
        self.add_text(
            "STRING THEORY LANDSCAPE EXPLORER",
            margin + panel_padding,
            margin + panel_padding,
            48.0,
            Colors::ACCENT,
        );

        // Subtitle
        self.add_text(
            "Searching for our universe's compactification",
            margin + panel_padding,
            margin + panel_padding + 60.0,
            27.0,
            [0.6, 0.6, 0.7, 1.0],
        );

        // Generation stats
        let stats_y = margin + panel_padding + 120.0;
        self.add_text(
            &format!("Generation: {}", generation),
            margin + panel_padding,
            stats_y,
            32.0,
            [0.9, 0.9, 0.9, 1.0],
        );
        self.add_text(
            &format!("Evaluated: {}", format_number(total_evaluated)),
            margin + panel_padding,
            stats_y + 45.0,
            32.0,
            [0.9, 0.9, 0.9, 1.0],
        );

        if let Some(stats) = history.last() {
            self.add_text(
                &format!("Stagnation: {} gen", stats.stagnation_generations),
                margin + panel_padding,
                stats_y + 90.0,
                29.0,
                if stats.stagnation_generations > 30 {
                    Colors::WARNING
                } else {
                    [0.9, 0.9, 0.9, 1.0]
                },
            );
            self.add_text(
                &format!("Landscape Collapses: {}", stats.landscape_collapses),
                margin + panel_padding,
                stats_y + 130.0,
                29.0,
                Colors::ACCENT,
            );
        }

        // Best fitness section
        let best_y = stats_y + 185.0;
        self.add_text(
            "BEST FITNESS",
            margin + panel_padding,
            best_y,
            29.0,
            Colors::ACCENT,
        );

        if let Some(best) = best_ever {
            self.add_text(
                &format!("{:.6e}", best.fitness),
                margin + panel_padding,
                best_y + 42.0,
                42.0,
                Colors::SUCCESS,
            );

            self.add_text(
                &format!("Fermion Generations: {}", best.physics.n_generations),
                margin + panel_padding,
                best_y + 96.0,
                27.0,
                if best.physics.n_generations == 3 {
                    Colors::SUCCESS
                } else {
                    Colors::ERROR
                },
            );
        }

        // Physical constants comparison
        let const_y = best_y + 145.0;
        self.add_text(
            "PHYSICAL CONSTANTS",
            margin + panel_padding,
            const_y,
            29.0,
            Colors::ACCENT,
        );

        if let Some(best) = best_ever {
            let predicted = best.physics.to_array();
            let labels = ["α_em", "α_s", "sin²θ_W", "m_e/M_Pl", "m_p/M_Pl", "Λ"];

            for (i, label) in labels.iter().enumerate() {
                let y = const_y + 48.0 + i as f32 * 67.0;
                let ratio = predicted[i] / TARGETS[i];
                let log_err = (ratio.log10()).abs();

                let status_color = if log_err < 0.1 {
                    Colors::SUCCESS
                } else if log_err < 1.0 {
                    Colors::WARNING
                } else {
                    Colors::ERROR
                };

                self.add_text(label, margin + panel_padding, y, 27.0, [0.7, 0.7, 0.8, 1.0]);

                // Smart formatting: decimal for reasonable values, scientific for extreme
                let pred_str = format_smart(predicted[i]);
                let target_str = format_smart(TARGETS[i]);

                self.add_text(
                    &pred_str,
                    margin + panel_padding + 145.0,
                    y,
                    24.0,
                    status_color,
                );

                self.add_text(
                    &format!("/ {}", target_str),
                    margin + panel_padding + 290.0,
                    y,
                    21.0,
                    [0.5, 0.5, 0.6, 1.0],
                );

                // Ratio indicator (truncate large values)
                let ratio_str = if ratio > 1e6 {
                    format!("×{:.0e}", ratio)
                } else if ratio > 1.0 {
                    format!("×{:.1}", ratio)
                } else if ratio < 1e-6 {
                    format!("÷{:.0e}", 1.0 / ratio)
                } else {
                    format!("÷{:.1}", 1.0 / ratio)
                };
                self.add_text(
                    &ratio_str,
                    margin + panel_padding + 450.0,
                    y,
                    21.0,
                    status_color,
                );
            }
        }

        // Scatter plot title
        self.add_text(
            "PARAMETER SPACE (g_s vs ln V)",
            scatter_x + panel_padding,
            margin + panel_padding,
            29.0,
            Colors::ACCENT,
        );

        // Scatter axes labels
        self.add_text("g_s = 0.01", grid_x, grid_y + grid_h + 13.0, 21.0, [0.5, 0.5, 0.6, 1.0]);
        self.add_text(
            "g_s = 0.5",
            grid_x + grid_w - 95.0,
            grid_y + grid_h + 13.0,
            21.0,
            [0.5, 0.5, 0.6, 1.0],
        );

        // Graph title
        self.add_text(
            "FITNESS HISTORY (log scale)",
            scatter_x + panel_padding,
            graph_y + panel_padding,
            29.0,
            Colors::ACCENT,
        );

        // Legend
        self.add_text(
            "● Best",
            scatter_x + scatter_w - 160.0,
            graph_y + panel_padding,
            24.0,
            Colors::SUCCESS,
        );
        self.add_text(
            "● Avg",
            scatter_x + scatter_w - 80.0,
            graph_y + panel_padding,
            24.0,
            [0.3, 0.5, 0.8, 1.0],
        );
    }

    fn add_text(&mut self, text: &str, x: f32, y: f32, size: f32, _color: [f32; 4]) {
        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size * 1.2));
        buffer.set_size(&mut self.font_system, Some(600.0), Some(size * 2.0));
        buffer.set_text(
            &mut self.font_system,
            text,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        buffer.shape_until_scroll(&mut self.font_system, false);

        self.text_buffers.push((buffer, x, y));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update viewport
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.size.width,
                height: self.size.height,
            },
        );

        // Prepare text areas from stored buffers with positions
        let text_areas: Vec<TextArea> = self.text_buffers
            .iter()
            .map(|(buffer, x, y)| TextArea {
                buffer,
                left: *x,
                top: *y,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.size.width as i32,
                    bottom: self.size.height as i32,
                },
                default_color: GlyphonColor::rgb(220, 220, 230),
                custom_glyphs: &[],
            })
            .collect();

        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: Colors::BACKGROUND[0] as f64,
                            g: Colors::BACKGROUND[1] as f64,
                            b: Colors::BACKGROUND[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw shapes
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);

            // Draw text
            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut render_pass)
                .unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// === Drawing Helpers ===

fn draw_panel(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    screen_w: f32,
    screen_h: f32,
) {
    // Background
    draw_rect(vertices, x, y, width, height, Colors::PANEL_BG, screen_w, screen_h);
    // Border
    draw_rect_outline(vertices, x, y, width, height, 1.5, Colors::PANEL_BORDER, screen_w, screen_h);
}

fn draw_rect(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) {
    let x1 = (x / screen_w) * 2.0 - 1.0;
    let y1 = 1.0 - (y / screen_h) * 2.0;
    let x2 = ((x + width) / screen_w) * 2.0 - 1.0;
    let y2 = 1.0 - ((y + height) / screen_h) * 2.0;

    vertices.push(Vertex { position: [x1, y1], color });
    vertices.push(Vertex { position: [x2, y1], color });
    vertices.push(Vertex { position: [x2, y2], color });

    vertices.push(Vertex { position: [x1, y1], color });
    vertices.push(Vertex { position: [x2, y2], color });
    vertices.push(Vertex { position: [x1, y2], color });
}

fn draw_rect_outline(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    thickness: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) {
    // Top
    draw_rect(vertices, x, y, width, thickness, color, screen_w, screen_h);
    // Bottom
    draw_rect(vertices, x, y + height - thickness, width, thickness, color, screen_w, screen_h);
    // Left
    draw_rect(vertices, x, y, thickness, height, color, screen_w, screen_h);
    // Right
    draw_rect(vertices, x + width - thickness, y, thickness, height, color, screen_w, screen_h);
}

fn draw_grid(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    h_lines: usize,
    v_lines: usize,
    screen_w: f32,
    screen_h: f32,
) {
    // Horizontal lines
    for i in 0..=v_lines {
        let ly = y + (i as f32 / v_lines as f32) * height;
        draw_line(vertices, x, ly, x + width, ly, 1.0, Colors::GRID_LINE, screen_w, screen_h);
    }
    // Vertical lines
    for i in 0..=h_lines {
        let lx = x + (i as f32 / h_lines as f32) * width;
        draw_line(vertices, lx, y, lx, y + height, 1.0, Colors::GRID_LINE, screen_w, screen_h);
    }
}

fn draw_line(
    vertices: &mut Vec<Vertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    thickness: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();

    if len < 0.001 {
        return;
    }

    let nx = -dy / len * thickness * 0.5;
    let ny = dx / len * thickness * 0.5;

    let to_ndc = |px: f32, py: f32| -> [f32; 2] {
        [(px / screen_w) * 2.0 - 1.0, 1.0 - (py / screen_h) * 2.0]
    };

    let p1 = to_ndc(x1 + nx, y1 + ny);
    let p2 = to_ndc(x1 - nx, y1 - ny);
    let p3 = to_ndc(x2 - nx, y2 - ny);
    let p4 = to_ndc(x2 + nx, y2 + ny);

    vertices.push(Vertex { position: p1, color });
    vertices.push(Vertex { position: p2, color });
    vertices.push(Vertex { position: p3, color });

    vertices.push(Vertex { position: p1, color });
    vertices.push(Vertex { position: p3, color });
    vertices.push(Vertex { position: p4, color });
}

fn draw_point(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    size: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) {
    let half = size * 0.5;
    draw_rect(vertices, x - half, y - half, size, size, color, screen_w, screen_h);
}

/// Convert fitness to color gradient
fn fitness_to_color(f: f32) -> [f32; 4] {
    let f = f.clamp(1e-15, 1.0);
    // Log scale: fitness ranges from ~1e-15 to ~1e-5 typically
    let t = ((f.log10() + 15.0) / 12.0).clamp(0.0, 1.0);

    if t < 0.33 {
        // Blue to Cyan
        let s = t / 0.33;
        [0.1, 0.2 + s * 0.6, 0.9, 0.8]
    } else if t < 0.66 {
        // Cyan to Yellow
        let s = (t - 0.33) / 0.33;
        [s * 0.9, 0.8, 0.9 - s * 0.7, 0.85]
    } else {
        // Yellow to Red
        let s = (t - 0.66) / 0.34;
        [0.9 + s * 0.1, 0.8 - s * 0.5, 0.2 - s * 0.1, 0.9]
    }
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
