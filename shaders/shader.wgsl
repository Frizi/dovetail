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
[[group(0), binding(4)]]
var<storage> visibility_buffer: [[access(read_write)]] VisBuffer;

[[group(1), binding(0)]]
var depth_pyramid: texture_2d<f32>;
[[group(1), binding(1)]]
var<uniform> pyramid_info: PyramidInfo;

fn frustum_cull_sphere(sphere: vec4<f32>) -> bool {
    var visible: bool = true;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        visible = visible &&
            dot(camera.frustum[i], vec4<f32>(sphere.xyz, 1.0)) > -sphere.w;
    }
    return visible;
}

struct SphereProj {
    accept: bool;
    aabb: vec4<f32>;
};

fn project_sphere(sphere: vec4<f32>, znear: f32, P00: f32, P11: f32) -> SphereProj {
	if (-sphere.z - znear < sphere.w) {
        return SphereProj(false, vec4<f32>(0.0));
    }

	let cx = -sphere.xz;
    let r = sphere.w;
	let vx = vec2<f32>(sqrt(dot(cx, cx) - r * r), r);
	let minx = mat2x2<f32>(vx, vec2<f32>(-vx.y, vx.x)) * cx;
	let maxx = mat2x2<f32>(vec2<f32>(vx.x, -vx.y), vx.yx) * cx;

	let cy = -sphere.yz;
	let vy = vec2<f32>(sqrt(dot(cy, cy) - r * r), r);
	let miny = mat2x2<f32>(vy, vec2<f32>(-vy.y, vy.x)) * cy;
	let maxy = mat2x2<f32>(vec2<f32>(vy.x, -vy.y), vy.yx) * cy;

	let aabb = vec4<f32>(
        minx.x / minx.y * P00,
        miny.x / miny.y * P11,
        maxx.x / maxx.y * P00,
        maxy.x / maxy.y * P11
    );
	let aabb = aabb.xwzy * vec4<f32>(-0.5, 0.5, -0.5, 0.5) + vec4<f32>(0.5); // clip space -> uv space
    return SphereProj(true, clamp(aabb, vec4<f32>(0.0), vec4<f32>(1.0)));
}



fn select_lod(mesh_index: u32, center: vec3<f32>) -> MeshLod {
    let eye_pos = camera.position;
    let lod_base = 14.0;
    let lod_step = 1.5;
    let lod_select = log2(distance(center, eye_pos) / lod_base) / log2(lod_step);
    let lod_index = min(u32(max(lod_select + 1.0, 0.0)), mesh_buffer.meshes[mesh_index].max_lod);
    return mesh_buffer.meshes[mesh_index].levels[lod_index];
}


[[stage(compute), workgroup_size(32)]]
fn cull_early(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    if (visibility_buffer.vis[global_id.x] == 0u) {
        cmd_buffer.commands[global_id.x] = DrawCmd(0u, 0u, 0u, 0u, 0u);
		return;
    }
    let draw = mesh_draw_buffer.mesh_draws[global_id.x];
    let mesh_index = draw.mesh_index;
    let mesh = mesh_buffer.meshes[mesh_index];

    // frustum culling
    let sphere = transform_sphere(mesh.bound_sphere, draw.position_scale, draw.orientation);
    let visible = frustum_cull_sphere(sphere);

    // select lod level
    var lod: MeshLod = select_lod(mesh_index, sphere.xyz);
    cmd_buffer.commands[global_id.x] = DrawCmd(
        lod.index_count,
        bool_to_uint(visible),
        lod.index_offset,
        mesh_buffer.meshes[mesh_index].vertex_offset,
        global_id.x,
    );
}

[[stage(compute), workgroup_size(32)]]
fn cull_late(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let draw = mesh_draw_buffer.mesh_draws[global_id.x];
    let mesh_index = draw.mesh_index;
    let mesh = mesh_buffer.meshes[mesh_index];

    // frustum culling
    let world_space_sphere = transform_sphere(mesh.bound_sphere, draw.position_scale, draw.orientation);
    let camera_space_sphere = vec4<f32>((camera.view * vec4<f32>(world_space_sphere.xyz, 1.0)).xyz, world_space_sphere.w);
    var visible: bool = frustum_cull_sphere(world_space_sphere);

    // occlusion culling using depth pyramid from early pass
    let znear = camera.projection.w.z;
    let projected = project_sphere(camera_space_sphere, znear, camera.projection.x.x, camera.projection.y.y);
    if (visible && projected.accept)
    {
        let dims = vec2<u32>(pyramid_info.width - 1u, pyramid_info.height - 1u);
        let aabb_xxyy = vec4<f32>(
            projected.aabb.xz * f32(dims.x),
            projected.aabb.yw * f32(dims.y),
        );
        let level = i32(log2(max(aabb_xxyy.x - aabb_xxyy.y, aabb_xxyy.z - aabb_xxyy.w)));
        let aabb_xxyy_i32 = vec4<i32>(aabb_xxyy);
        let sample_pos = vec4<i32>(
            aabb_xxyy_i32.x >> u32(level + 1),
            aabb_xxyy_i32.y >> u32(level + 1),
            aabb_xxyy_i32.z >> u32(level + 1),
            aabb_xxyy_i32.w >> u32(level + 1),
        );

        let depth0 = textureLoad(depth_pyramid, sample_pos.xz, level).x;
        let depth1 = textureLoad(depth_pyramid, sample_pos.xw, level).x;
        let depth2 = textureLoad(depth_pyramid, sample_pos.yz, level).x;
        let depth3 = textureLoad(depth_pyramid, sample_pos.yw, level).x;
        let depth_min = min(min(depth0, depth1), min(depth2, depth3));
        let depth_sphere = znear / (-camera_space_sphere.z - camera_space_sphere.w);
        visible = depth_sphere > depth_min;
    }

    // select lod level
    var lod: MeshLod = select_lod(mesh_index, world_space_sphere.xyz);
    cmd_buffer.commands[global_id.x] = DrawCmd(
        lod.index_count,
        bool_to_uint(visible && visibility_buffer.vis[global_id.x] == 0u),
        lod.index_offset,
        mesh_buffer.meshes[mesh_index].vertex_offset,
        global_id.x,
    );
    visibility_buffer.vis[global_id.x] = bool_to_uint(visible);
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
    return vec4<f32>(rgb, 0.3);
}
