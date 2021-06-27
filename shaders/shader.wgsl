#include "_structs.wgsl";
#include "_util.wgsl";

[[group(0), binding(0)]]
var<uniform> camera: Camera;
[[group(0), binding(1)]]
var<storage> mesh_buffer: [[access(read)]] MeshBuffer;
[[group(0), binding(2)]]
var<storage> mesh_draw_buffer: [[access(read)]] MeshDrawBuffer;
[[group(0), binding(3)]]
var<storage> cmd_buffer: [[access(write)]] CmdBuffer;

fn frustum_cull_sphere(sphere: vec4<f32>) -> bool {
    var visible: bool = true;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        visible = visible &&
            dot(camera.frustum[i], vec4<f32>(sphere.xyz, 1.0)) > -sphere.w;
    }
    return visible;
}

[[stage(compute), workgroup_size(32)]]
fn emit_draws(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let draw = mesh_draw_buffer.mesh_draws[global_id.x];
    let mesh_index = draw.mesh_index;
    let mesh = mesh_buffer.meshes[mesh_index];

    var visible: bool = true;

    // frustum culling
    let sphere = transform_sphere(mesh.bound_sphere, draw.position_scale, draw.orientation);
    visible = visible && frustum_cull_sphere(sphere);

    // select lod level
    let eye_pos = camera.position;
    let lod_base = 14.0;
    let lod_step = 1.5;
    let lod_select = log2(distance(sphere.xyz, eye_pos) / lod_base) / log2(lod_step);
    let lod_index = min(u32(max(lod_select + 1.0, 0.0)), mesh.max_lod);
    let lod = mesh_buffer.meshes[mesh_index].levels[lod_index];

    cmd_buffer.commands[global_id.x] = DrawCmd(
        lod.index_count,
        bool_to_uint(visible),
        lod.index_offset,
        mesh.vertex_offset,
        global_id.x,
    );
}

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(1), interpolate(flat)]] instance_index: u32;
    [[location(2), interpolate(flat)]] vertex_index: u32;
};

[[stage(vertex)]]
fn visibility_vs(
    [[builtin(instance_index)]] instance_index: u32,
    [[builtin(vertex_index)]] vertex_index: u32,
    [[location(0)]] position: vec3<f32>,
) -> VertexOutput {
    let mesh_draw = mesh_draw_buffer.mesh_draws[instance_index];
    var pos: vec3<f32> = position;
    pos = rotate_quat(pos * mesh_draw.position_scale.w, mesh_draw.orientation) + mesh_draw.position_scale.xyz;
    return VertexOutput(
        camera.view_proj * vec4<f32>(pos, 1.0),
        instance_index,
        vertex_index,
    );
}

[[stage(fragment)]]
fn visibility_fs(input: VertexOutput) -> [[location(0)]] vec4<f32> {
    let rgb1 = hash_rgb(input.instance_index);
    let rgb2 = hash_rgb(input.vertex_index);
    let rgb = mix(rgb1, rgb2, vec3<f32>(0.3));
    return vec4<f32>(rgb, 1.0);
}
