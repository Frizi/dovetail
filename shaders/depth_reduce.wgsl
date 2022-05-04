#include "_structs.wgsl";

[[group(0), binding(0)]]
var image_src: texture_2d<f32>;
[[group(0), binding(1)]]
var image_dst: texture_storage_2d<r32float, write>;
[[group(0), binding(2)]]
var<uniform> pyramid_info: PyramidInfo;

struct PushConstants {
    mip_level: u32;
};

var<push_constant> push: PushConstants;

[[stage(compute), workgroup_size(16, 16)]]
fn depth_reduce(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    var id2x: vec2<i32> = vec2<i32>(global_id.xy) * 2;
    // let max_size = vec2<i32>(i32(pyramid_info.width) - 1, i32(pyramid_info.height) - 1);
    let max_size = vec2<i32>(i32(pyramid_info.width >> push.mip_level) - 1, i32(pyramid_info.height >> push.mip_level) - 1);

    // Ideally this would use a single sample with min reduction, but that's not supported on wgpu right now
    let value0 = textureLoad(image_src, min(id2x, max_size), 0).x;
    let value1 = textureLoad(image_src, min(id2x + vec2<i32>(0, 1), max_size), 0).x;
    let value2 = textureLoad(image_src, min(id2x + vec2<i32>(1, 0), max_size), 0).x;
    let value3 = textureLoad(image_src, min(id2x + vec2<i32>(1, 1), max_size), 0).x;
    let value = min(min(value0, value1), min(value2, value3));

    textureStore(image_dst, vec2<i32>(global_id.xy), vec4<f32>(value));
}
