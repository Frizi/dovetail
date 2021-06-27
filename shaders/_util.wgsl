fn rotate_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32>
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn transform_sphere(sphere: vec4<f32>, position_scale: vec4<f32>, orientation: vec4<f32>) -> vec4<f32> {
    let center = rotate_quat(sphere.xyz, orientation) * position_scale.w + position_scale.xyz;
    let radius = sphere.w * position_scale.w;
    return vec4<f32>(center, radius);
}

fn bool_to_uint(visible: bool) -> u32 {
    if (visible) { return 1u; }
    return 0u;
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
