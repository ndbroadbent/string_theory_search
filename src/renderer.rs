//! GPU-accelerated visualization of the string landscape search.
//!
//! Displays:
//! - Population as points in a 2D projection of parameter space
//! - Fitness as color (blue = low, red = high)
//! - History of best fitness over time
//! - Current best individual's physics

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use winit::window::Window;

use crate::fitness::Individual;
use crate::genetic::GenerationStats;

/// Vertex for rendering points
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
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
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

        // Shader for rendering colored points
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
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Initial vertex buffer (will be updated each frame)
        let max_vertices = 100_000;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

    /// Update visualization with current population and history
    pub fn update(&mut self, population: &[Individual], history: &[GenerationStats]) {
        let mut vertices = Vec::with_capacity(self.max_vertices);

        // === Draw population as points ===
        // Project high-dimensional genome onto 2D using first two principal components
        // (simplified: use specific meaningful parameters)
        for ind in population.iter().take(2000) {
            // X: string coupling (0 to 0.5 -> -0.9 to 0.9)
            let x = (ind.genome.string_coupling * 4.0 - 1.0) as f32 * 0.4 - 0.45;

            // Y: log(CY volume) normalized
            let y = ((ind.genome.cy_volume.ln() - 1.0) / 5.0) as f32 * 0.4 + 0.3;

            // Color based on fitness (blue -> green -> yellow -> red)
            let f = ind.fitness as f32;
            let color = fitness_to_color(f);

            // Draw as small quad (two triangles)
            let size = 0.004 + f * 0.008; // Larger points for higher fitness
            add_quad(&mut vertices, x, y, size, color);
        }

        // === Draw fitness history as line graph ===
        if history.len() > 1 {
            let graph_left: f32 = 0.1;
            let graph_right: f32 = 0.95;
            let graph_bottom: f32 = -0.95;
            let graph_top: f32 = -0.5;

            // Background
            add_quad(
                &mut vertices,
                (graph_left + graph_right) / 2.0,
                (graph_bottom + graph_top) / 2.0,
                0.45,
                [0.1, 0.1, 0.15, 0.8],
            );

            // Plot best fitness over time
            let max_points = 500;
            let step = (history.len() / max_points).max(1);
            let points: Vec<_> = history.iter().step_by(step).collect();

            for i in 0..points.len().saturating_sub(1) {
                let x1 = graph_left + (i as f32 / points.len() as f32) * (graph_right - graph_left);
                let x2 =
                    graph_left + ((i + 1) as f32 / points.len() as f32) * (graph_right - graph_left);

                // Log scale for fitness display
                let y1 = graph_bottom
                    + ((points[i].best_fitness.ln().max(-20.0) + 20.0) / 20.0) as f32
                        * (graph_top - graph_bottom);
                let y2 = graph_bottom
                    + ((points[i + 1].best_fitness.ln().max(-20.0) + 20.0) / 20.0) as f32
                        * (graph_top - graph_bottom);

                // Draw line segment as thin quad
                add_line(&mut vertices, x1, y1, x2, y2, 0.003, [0.2, 1.0, 0.3, 1.0]);
            }

            // Average fitness in different color
            for i in 0..points.len().saturating_sub(1) {
                let x1 = graph_left + (i as f32 / points.len() as f32) * (graph_right - graph_left);
                let x2 =
                    graph_left + ((i + 1) as f32 / points.len() as f32) * (graph_right - graph_left);

                let y1 = graph_bottom
                    + ((points[i].avg_fitness.ln().max(-20.0) + 20.0) / 20.0) as f32
                        * (graph_top - graph_bottom);
                let y2 = graph_bottom
                    + ((points[i + 1].avg_fitness.ln().max(-20.0) + 20.0) / 20.0) as f32
                        * (graph_top - graph_bottom);

                add_line(&mut vertices, x1, y1, x2, y2, 0.002, [0.5, 0.5, 1.0, 0.7]);
            }
        }

        // === Draw parameter space axes labels area ===
        // Left panel background
        add_quad(&mut vertices, -0.75, 0.0, 0.22, [0.1, 0.1, 0.15, 0.8]);

        // Truncate if too many vertices
        if vertices.len() > self.max_vertices {
            vertices.truncate(self.max_vertices);
        }

        self.num_vertices = vertices.len() as u32;

        // Upload to GPU
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Convert fitness value to color (blue -> cyan -> green -> yellow -> red)
fn fitness_to_color(f: f32) -> [f32; 4] {
    let f = f.clamp(0.0, 1.0);

    // Use log scale for better visualization of small fitness differences
    let t = (f.ln().max(-10.0) + 10.0) / 10.0;

    if t < 0.25 {
        // Blue to cyan
        let s = t / 0.25;
        [0.0, s * 0.8, 1.0, 0.7]
    } else if t < 0.5 {
        // Cyan to green
        let s = (t - 0.25) / 0.25;
        [0.0, 0.8 + s * 0.2, 1.0 - s, 0.7]
    } else if t < 0.75 {
        // Green to yellow
        let s = (t - 0.5) / 0.25;
        [s, 1.0, 0.0, 0.8]
    } else {
        // Yellow to red
        let s = (t - 0.75) / 0.25;
        [1.0, 1.0 - s * 0.8, 0.0, 0.9]
    }
}

/// Add a quad (two triangles) centered at (x, y)
fn add_quad(vertices: &mut Vec<Vertex>, x: f32, y: f32, half_size: f32, color: [f32; 4]) {
    // Triangle 1
    vertices.push(Vertex {
        position: [x - half_size, y - half_size],
        color,
    });
    vertices.push(Vertex {
        position: [x + half_size, y - half_size],
        color,
    });
    vertices.push(Vertex {
        position: [x + half_size, y + half_size],
        color,
    });

    // Triangle 2
    vertices.push(Vertex {
        position: [x - half_size, y - half_size],
        color,
    });
    vertices.push(Vertex {
        position: [x + half_size, y + half_size],
        color,
    });
    vertices.push(Vertex {
        position: [x - half_size, y + half_size],
        color,
    });
}

/// Add a line segment as a thin quad
fn add_line(
    vertices: &mut Vec<Vertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    thickness: f32,
    color: [f32; 4],
) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();

    if len < 0.0001 {
        return;
    }

    // Perpendicular direction
    let nx = -dy / len * thickness;
    let ny = dx / len * thickness;

    vertices.push(Vertex {
        position: [x1 + nx, y1 + ny],
        color,
    });
    vertices.push(Vertex {
        position: [x1 - nx, y1 - ny],
        color,
    });
    vertices.push(Vertex {
        position: [x2 - nx, y2 - ny],
        color,
    });

    vertices.push(Vertex {
        position: [x1 + nx, y1 + ny],
        color,
    });
    vertices.push(Vertex {
        position: [x2 - nx, y2 - ny],
        color,
    });
    vertices.push(Vertex {
        position: [x2 + nx, y2 + ny],
        color,
    });
}
