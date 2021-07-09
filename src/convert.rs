use glam::Vec4;

#[derive(Clone, Copy)]
struct Vertex {
    position: Vec4,
}

fn main() {
    tracing_subscriber::fmt::init();

    let path: String = std::env::args().skip(1).next().expect("mesh path required");

    let reader =
        std::io::BufReader::new(std::fs::File::open(path).expect("Failed to open mesh file"));
    let model: obj::Obj<obj::Position, u32> = obj::load_obj(reader).expect("Failed to parse obj");

    let vertex_count = model.vertices.len() as u32;
    let vertices: Vec<Vertex> = model
        .vertices
        .into_iter()
        .map(|v| Vertex {
            position: Vec4::new(v.position[0], v.position[1], v.position[2], 1.0),
        })
        .collect();

    let adapter = meshopt::VertexDataAdapter::new(
        vert_bytes,
        std::mem::size_of::<Vertex>(),
        memoffset::offset_of!(Vertex, position),
    )
    .expect("Failed to create vertex data adapter");

    let opt_indices = meshopt::optimize_vertex_cache(&model.indices, vertices.len());
    let bound_sphere = bounding_sphere(&vertices);
}
