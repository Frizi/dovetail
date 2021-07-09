struct MeshDraw {
    position_scale: vec4<f32>;
    orientation: vec4<f32>;
    mesh_index: u32;
};

struct MeshLod {
    index_offset: u32;
    index_count: u32;
};

struct Mesh {
    bound_sphere: vec4<f32>;
    vertex_offset: u32;
    vertex_count: u32;
    max_lod: u32;
    levels: array<MeshLod, 8>;
};

struct DrawCmdIndexed {
    index_count: u32;
    instance_count: u32;
    first_index: u32;
    base_vertex: u32;
    first_instance: u32;
};

struct DrawCmd {
    vertex_count: u32;
    instance_count: u32;
    base_vertex: u32;
    base_instance: u32;
};

[[block]]
struct MeshBuffer {
    meshes: array<Mesh>;
};

[[block]]
struct MeshDrawBuffer {
    mesh_draws: array<MeshDraw>;
};

[[block]]
struct CmdBuffer {
    commands: array<DrawCmd>;
};

[[block]]
struct VisBuffer {
    vis: array<u32>;
};

[[block]]
struct Camera {
    view: mat4x4<f32>;
    projection: mat4x4<f32>;
    view_proj: mat4x4<f32>;
    frustum: array<vec4<f32>, 4>;
    position: vec3<f32>;
};

[[block]]
struct PyramidInfo {
    width: u32;
    height: u32;
};