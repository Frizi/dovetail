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
    levels: array<MeshLod, 8>;
};

struct DrawCmd {
    index_count: u32;
    instance_count: u32;
    first_index: u32;
    base_vertex: u32;
    first_instance: u32;
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

fn rotate_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32>
{
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

[[group(0), binding(0)]]
var<storage> mesh_buffer: [[access(read)]] MeshBuffer;
[[group(0), binding(1)]]
var<storage> mesh_draw_buffer: [[access(read)]] MeshDrawBuffer;
[[group(0), binding(2)]]
var<storage> cmd_buffer: [[access(write)]] CmdBuffer;

[[stage(compute), workgroup_size(32)]]
fn emit_draws(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let mesh_index = mesh_draw_buffer.mesh_draws[global_id.x].mesh_index;
    // TODO: cull and select lod
    let mesh = mesh_buffer.meshes[mesh_index];
    let lod = mesh.levels[0];
    
    cmd_buffer.commands[global_id.x] = DrawCmd(
        lod.index_count,
        1u,
        lod.index_offset,
        mesh.vertex_offset,
        global_id.x,
    );
}

fn hash(x: u32) -> u32 {
    let x = x + ( x << 10u );
    let x = x ^ ( x >>  6u );
    let x = x + ( x <<  3u );
    let x = x ^ ( x >> 11u );
    let x = x + ( x << 15u );
    return x;
}

fn hash_rgb(x: u32) -> vec3<f32> {
    let x: u32 = hash(x);
    let r = (x >> 8u) & 255u;
    let g = (x >> 4u) & 255u;
    let b = (x >> 0u) & 255u;
    return vec3<f32>(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0);
}

// TODO: 
// - figure out how to render multiple mesh types using single draw indirect call
// - do frustum culling on cpu
// - do occlusion culling on gpu using last-frame visibility as a starting approximation

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(1), interpolate(flat)]] instance_index: u32;
};

[[stage(vertex)]]
fn visibility_vs(
    [[builtin(instance_index)]] instance_index: u32,
    [[location(0)]] position: vec3<f32>,
) -> VertexOutput {
    let mesh_draw = mesh_draw_buffer.mesh_draws[instance_index];
    var pos: vec3<f32> = position;
    pos = rotate_quat(pos * mesh_draw.position_scale.w, mesh_draw.orientation) + mesh_draw.position_scale.xyz;
    pos = vec3<f32>(pos.xy * 0.05, pos.z * 0.01 + 0.1);
    return VertexOutput(
        vec4<f32>(pos * 5.0, 1.0),
        instance_index,
    );
}

[[stage(fragment)]]
fn visibility_fs(input: VertexOutput) -> [[location(0)]] vec4<f32> {
    let rgb = hash_rgb(input.instance_index);
    return vec4<f32>(rgb, 1.0);
}
