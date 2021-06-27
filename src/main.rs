use std::{f32::consts::PI, fmt::Display, path::Path};

use glam::{Mat4, Quat, Vec3, Vec4};
use tasks::Spawner;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod shader;
mod tasks;
mod util;

use shader::*;
use util::*;

type Vertex = obj::Position;

fn load_mesh(
    path: impl AsRef<Path>,
    index_buffer: &mut Vec<u32>,
    vertex_buffer: &mut Vec<Vertex>,
) -> Mesh {
    let reader =
        std::io::BufReader::new(std::fs::File::open(path).expect("Failed to open mesh file"));
    let model: obj::Obj<Vertex, u32> = obj::load_obj(reader).expect("Failed to parse obj");

    let vertex_offset = vertex_buffer.len() as u32;
    let vertex_count = model.vertices.len() as u32;
    let vert_bytes = as_bytes(&model.vertices);

    let adapter = meshopt::VertexDataAdapter::new(
        vert_bytes,
        std::mem::size_of::<Vertex>(),
        memoffset::offset_of!(Vertex, position),
    )
    .expect("Failed to create vertex data adapter");

    let opt_indices = meshopt::optimize_vertex_cache(&model.indices, model.vertices.len());
    let bound_sphere = bounding_sphere(&model.vertices);

    const LOD_INIT: MeshLod = MeshLod {
        index_count: 0,
        index_offset: 0,
    };

    let mut levels = [LOD_INIT; 8];

    levels[0] = MeshLod {
        index_offset: index_buffer.len() as u32,
        index_count: opt_indices.len() as u32,
    };

    vertex_buffer.extend_from_slice(&model.vertices);
    index_buffer.extend_from_slice(&opt_indices);
    // index_buffer.extend(opt_indices.iter().map(|i| i + index_base));

    let mut indices_target = opt_indices.len() * 3 / 4;
    let mut simplified_indices = Vec::new();

    for level in 1..8 {
        let indices = match level {
            1 => &opt_indices[..],
            _ => &simplified_indices[..],
        };

        simplified_indices = meshopt::simplify(indices, &adapter, indices_target, 1e-2);
        meshopt::optimize_vertex_cache_in_place(&simplified_indices, model.vertices.len());

        indices_target = simplified_indices.len() * 3 / 4;
        let index_offset = index_buffer.len() as u32;
        let index_count = simplified_indices.len() as u32;
        index_buffer.extend_from_slice(&simplified_indices);

        levels[level] = MeshLod {
            index_count,
            index_offset,
        };
    }

    Mesh {
        bound_sphere,
        vertex_offset,
        vertex_count,
        levels,
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
struct MeshDraw {
    position_scale: Vec4,
    orientation: Quat,
    mesh_index: u32,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
struct Mesh {
    bound_sphere: Vec4,
    vertex_offset: u32,
    vertex_count: u32,
    levels: [MeshLod; 8],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Camera {
    view: Mat4,
    projection: Mat4,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
struct MeshLod {
    index_offset: u32,
    index_count: u32,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let spawner = Spawner::new();
    let instance = wgpu::Instance::new(wgpu::BackendBit::all());
    let surface = unsafe { instance.create_surface(&window) };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MULTI_DRAW_INDIRECT | wgpu::Features::TIMESTAMP_QUERY,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let swapchain_format = adapter.get_swap_chain_preferred_format(&surface).unwrap();

    let size = window.inner_size();
    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let mut depth_desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
    };

    let depth_view_desc = wgpu::TextureViewDescriptor {
        label: None,
        format: Some(wgpu::TextureFormat::Depth32Float),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::DepthOnly,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    };

    let mut depth_image = device.create_texture(&depth_desc);
    let mut depth_image_view = depth_image.create_view(&depth_view_desc);

    let mut vertex_data = Vec::new();
    let mut index_data = Vec::new();

    let meshes: Vec<Mesh> = std::env::args()
        .skip(1)
        .map(|path| load_mesh(path, &mut index_data, &mut vertex_data))
        .collect();

    let mut camera = Camera {
        view: Mat4::IDENTITY,
        projection: Mat4::IDENTITY,
    };

    let mut camera_quat = Quat::IDENTITY;
    let mut camera_pos = Vec3::new(0.0, 0.0, 10.0);
    let mut camera_yaw = 0.0;
    let mut camera_pitch = 0.0;
    let mut mouse_pos = None;

    let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        contents: as_bytes(std::slice::from_ref(&camera)),
    });

    let mesh_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsage::STORAGE,
        contents: as_bytes(&meshes),
    });

    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsage::INDEX,
        contents: as_bytes(&index_data),
    });
    drop(index_data);

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsage::VERTEX,
        contents: as_bytes(&vertex_data),
    });
    drop(vertex_data);

    let mut mesh_draws: Vec<MeshDraw> = Vec::new();

    for i in 0..32 {
        mesh_draws.push(MeshDraw {
            position_scale: Vec4::new(
                0.5 * (i as f32 * 142.33).sin(),
                0.5 * (i as f32 * 27.43).cos(),
                0.0,
                0.5 + 0.3 * (i as f32 * 1234.77).sin(),
            ),
            orientation: Quat::from_axis_angle(
                Vec3::new(1.0, 0.0, 1.0).normalize(),
                i as f32 * 12.763,
            ),
            mesh_index: i % meshes.len() as u32,
        });
    }

    let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT,
        size: mesh_draws.len() as u64 * std::mem::size_of::<[u32; 5]>() as u64,
        mapped_at_creation: false,
    });

    let mesh_draw_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsage::STORAGE,
        contents: as_bytes(&mesh_draws),
    });

    let num_mesh_draws = mesh_draws.len() as u32;
    drop(mesh_draws);

    // Load the shaders from disk
    let shader = load_shader(&device, "shader.wgsl");

    let visibility_shader_set: ShaderSet =
        vec![shader.entry("visibility_vs"), shader.entry("visibility_fs")].into();

    let emit_draws_shader_set: ShaderSet = vec![shader.entry("emit_draws")].into();

    let mut comp_bind_group_template = emit_draws_shader_set.pipeline_template(&device);
    let comp_bind_group = comp_bind_group_template
        .bind("mesh_buffer", &mesh_buffer)
        .bind("mesh_draw_buffer", &mesh_draw_buffer)
        .bind("cmd_buffer", &draw_commands_buffer)
        .build_bind_group(&device, 0);

    let mut draw_bind_group_template = visibility_shader_set.pipeline_template(&device);
    let draw_bind_group = draw_bind_group_template
        .bind("camera", &camera_buffer)
        .bind("mesh_draw_buffer", &mesh_draw_buffer)
        .build_bind_group(&device, 0);

    let comp_pipeline = comp_bind_group_template.compute_pipeline(&device, &emit_draws_shader_set);
    let render_pipeline = draw_bind_group_template.graphics_pipeline(
        &device,
        &visibility_shader_set,
        &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        }],
        &[swapchain_format.into()],
        Some(wgpu::DepthStencilState {
            format: depth_desc.format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil: Default::default(),
            bias: Default::default(),
        }),
    );

    let clear_color = wgpu::Color {
        r: 0.01,
        g: 0.02,
        b: 0.02,
        a: 1.0,
    };

    let mut use_indirect_draw = true;

    let mut staging_belt = wgpu::util::StagingBelt::new(0x100);

    // let num_timestamps = 3;
    // let timestamps = device.create_query_set(&wgpu::QuerySetDescriptor {
    //     ty: wgpu::QueryType::Timestamp,
    //     count: num_timestamps,
    // });

    // let timestamp_buffers = (0..4)
    //     .map(|_| {
    //         device.create_buffer(&wgpu::BufferDescriptor {
    //             label: None,
    //             usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
    //             size: num_timestamps as u64 * std::mem::size_of::<u64>() as u64,
    //             mapped_at_creation: false,
    //         })
    //     })
    //     .collect::<Vec<_>>();

    // let ts_period = queue.get_timestamp_period();

    let mut last_frame_time = std::time::Instant::now();
    let mut move_input = [false; 6];

    event_loop.run(move |evt, _, flow| match evt {
        Event::WindowEvent {
            event, window_id, ..
        } if window_id == window.id() => {
            if wants_close(&event) {
                *flow = ControlFlow::Exit;
                return;
            }
            match event {
                WindowEvent::Resized(size) => {
                    sc_desc.width = size.width;
                    sc_desc.height = size.height;
                    depth_desc.size.width = size.width;
                    depth_desc.size.height = size.height;

                    swap_chain = device.create_swap_chain(&surface, &sc_desc);
                    depth_image = device.create_texture(&depth_desc);
                    depth_image_view = depth_image.create_view(&depth_view_desc);
                }
                WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                    Some(
                        key
                        @
                        (winit::event::VirtualKeyCode::W
                        | winit::event::VirtualKeyCode::S
                        | winit::event::VirtualKeyCode::A
                        | winit::event::VirtualKeyCode::D
                        | winit::event::VirtualKeyCode::Q
                        | winit::event::VirtualKeyCode::E),
                    ) => {
                        let key_index = match key {
                            winit::event::VirtualKeyCode::W => 0,
                            winit::event::VirtualKeyCode::S => 1,
                            winit::event::VirtualKeyCode::D => 2,
                            winit::event::VirtualKeyCode::A => 3,
                            winit::event::VirtualKeyCode::Q => 4,
                            winit::event::VirtualKeyCode::E => 5,
                            _ => unreachable!(),
                        };

                        move_input[key_index] = match input.state {
                            winit::event::ElementState::Pressed => true,
                            winit::event::ElementState::Released => false,
                        };
                    }

                    Some(winit::event::VirtualKeyCode::I) => {
                        use_indirect_draw = !use_indirect_draw;
                    }
                    _ => {}
                },
                WindowEvent::CursorMoved { position, .. } => {
                    let old_pos = match mouse_pos.replace(position) {
                        Some(pos) => pos,
                        None => return,
                    };
                    let delta_x = position.x - old_pos.x;
                    let delta_y = position.y - old_pos.y;
                    camera_yaw -= (delta_x * 0.002) as f32;
                    camera_pitch = (camera_pitch + (-delta_y * 0.002) as f32).clamp(-PI, PI);
                    camera_quat =
                        Quat::from_euler(glam::EulerRot::YXZ, camera_yaw, camera_pitch, 0.0);
                }
                _ => {}
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawEventsCleared => {
            device.poll(wgpu::Maintain::Poll);
            spawner.run_until_stalled();
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            let frame = swap_chain
                .get_current_frame()
                .expect("Failed to acquire next swap chain texture")
                .output;

            let cpu_time_start = std::time::Instant::now();

            let delta_time = cpu_time_start.duration_since(last_frame_time);
            last_frame_time = cpu_time_start;

            let move_input_vec: Vec3 = move_input
                .iter()
                .enumerate()
                .map(|(i, active)| match (i, active) {
                    (0, true) => -Vec3::Z,
                    (1, true) => Vec3::Z,
                    (2, true) => Vec3::X,
                    (3, true) => -Vec3::X,
                    (4, true) => -Vec3::Y,
                    (5, true) => Vec3::Y,
                    _ => Vec3::ZERO,
                })
                .fold(Vec3::ZERO, |a, b| a + b);

            camera_pos += camera_quat * (move_input_vec * 5.0 * delta_time.as_secs_f32());
            camera.view =
                Mat4::from_scale_rotation_translation(Vec3::splat(1.0), camera_quat, camera_pos)
                    .inverse();
            camera.projection = Mat4::perspective_infinite_reverse_rh(
                1.3,
                window.inner_size().width as f32 / window.inner_size().height as f32,
                0.1,
            );
            dbg!(&camera_pos);

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // update camera uniform
            staging_belt
                .write_buffer(
                    &mut encoder,
                    &camera_buffer,
                    0,
                    wgpu::BufferSize::new(std::mem::size_of::<Camera>() as wgpu::BufferAddress)
                        .unwrap(),
                    &device,
                )
                .copy_from_slice(as_bytes(std::slice::from_ref(&camera)));

            staging_belt.finish();
            // encode passes
            // encoder.write_timestamp(&timestamps, 0);
            {
                let mut comp_pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                comp_pass.set_pipeline(&comp_pipeline);
                comp_pass.set_bind_group(0, &comp_bind_group, &[]);
                comp_pass.dispatch((num_mesh_draws + 31) / 32, 1, 1);
            }

            // encoder.write_timestamp(&timestamps, 1);

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &frame.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear_color),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_image_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });
                render_pass.set_pipeline(&render_pipeline);
                render_pass.set_bind_group(0, &draw_bind_group, &[]);
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.multi_draw_indexed_indirect(&draw_commands_buffer, 0, num_mesh_draws);
            }

            // encoder.write_timestamp(&timestamps, 1);

            // encoder.resolve_query_set(&timestamps, 0..num_timestamps, timestamp_buffer, 0);
            queue.submit(Some(encoder.finish()));
            drop(frame);

            spawner.spawn_local(staging_belt.recall());

            let cpu_time = cpu_time_start.elapsed();

            // let buf_slice = timestamp_buffer.slice(..);
            // let _ = buf_slice.map_async(wgpu::MapMode::Read);
            // device.poll(wgpu::Maintain::Poll);
            // let timestamps = buf_slice.get_mapped_range();
            // let gpu_time_compute = (timestamps[1] - timestamps[0]) as f64 * ts_period as f64;
            // let gpu_time_draw = (timestamps[2] - timestamps[1]) as f64 * ts_period as f64;
            // let gpu_time_total = (timestamps[2] - timestamps[0]) as f64 * ts_period as f64;
            // drop(timestamps);
            // timestamp_buffer.unmap();

            let title = format!(
                "dovetail :: CPU time: {}, indirect = {:?}",
                // DisplayTime::from_ns(gpu_time_total),
                // DisplayTime::from_ns(gpu_time_compute),
                // DisplayTime::from_ns(gpu_time_draw),
                DisplayTime(cpu_time),
                use_indirect_draw
            );
            window.set_title(&title);
            *flow = ControlFlow::Poll;
        }
        _ => *flow = ControlFlow::Poll,
    });
}

struct DisplayTime(std::time::Duration);
// impl DisplayTime {
//     fn from_ns(nanos: f64) -> Self {
//         Self(std::time::Duration::from_nanos(nanos as u64))
//     }
// }

impl Display for DisplayTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.0;
        if d.as_secs() >= 1 {
            write!(f, "{:.2}s", d.as_millis() as f32 / 1000.0)
        } else if d.as_millis() >= 1 {
            write!(f, "{:.2}ms", d.as_micros() as f32 / 1000.0)
        } else if d.as_micros() >= 1 {
            write!(f, "{:.2}µs", d.as_nanos() as f32 / 1000.0)
        } else {
            write!(f, "{}ns", d.as_nanos())
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("dovetail")
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run(event_loop, window));
}

fn wants_close(evt: &WindowEvent) -> bool {
    match evt {
        WindowEvent::CloseRequested => true,
        WindowEvent::KeyboardInput { input, .. } => {
            matches! { input.virtual_keycode, Some(winit::event::VirtualKeyCode::Escape)}
        }
        _ => false,
    }
}

fn bounding_sphere(vertices: &[Vertex]) -> Vec4 {
    let point_x = Vec3::from(vertices[0].position);
    // Find point y, the point furthest from point x
    let point_y = vertices.iter().fold(point_x, |acc, x| {
        let x = Vec3::from(x.position);
        if x.distance(point_x) >= acc.distance(point_x) {
            x
        } else {
            acc
        }
    });
    // Find point z, the point furthest from point y
    let point_z = vertices.iter().fold(point_y, |acc, x| {
        let x = Vec3::from(x.position);
        if x.distance(point_y) >= acc.distance(point_y) {
            x
        } else {
            acc
        }
    });
    // Construct a bounding sphere using these two points as the poles
    let mut origin = point_y.lerp(point_z, 0.5);
    let mut radius = point_y.distance(point_z) / 2.0;
    // Iteratively adjust sphere until it encloses all points
    loop {
        // Find the furthest point from the origin
        let point_n = vertices.iter().fold(point_x, |acc, x| {
            let x = Vec3::from(x.position);
            if x.distance(origin) >= acc.distance(origin) {
                x
            } else {
                acc
            }
        });
        // If the furthest point is outside the sphere, we need to adjust it
        let point_dist = point_n.distance(origin);
        if point_dist > radius {
            let radius_new = (radius + point_dist) / 2.0;
            let lerp_ratio = (point_dist - radius_new) / point_dist;
            origin = origin.lerp(point_n, lerp_ratio);
            radius = radius_new;
        } else {
            return origin.extend(radius);
        }
    }
}
