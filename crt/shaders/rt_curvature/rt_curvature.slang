#version 450

/*
    Ray traced curvature v1.0 by fishku
    Copyright (C) 2024
    Public domain license (CC0)

    This example demoes the following:
    - How to generate rays for tracing against the screen surface in the vertex
      shader.
    - How to trace the rays in the fragment shader.
    - How to achieve high-quality sampling with the found UV.

    Changelog:
    v1.0: Initial release.
*/

// If uncommented, pixel art is sampled sharply, better for stand-alone usage.
// If commented out, no coordinate distortion is done, which is better for
// appending. Enabling this option may give stronger aliasing artifacts when
// appending to other presets.
#define SHARP_PIXEL_SAMPLING

#include "../../../misc/shaders/input_transform/rotation.inc"
#include "parameters.inc"
#include "shared.inc"

layout(push_constant) uniform Push {
    vec4 OriginalSize;
    vec4 SourceSize;
    vec4 OutputSize;
    uint Rotation;
    uint FrameCount;
    float RT_CURV_ASPECT_H;
    float RT_CURV_ASPECT_V;
    float RT_CURV_SHAPE;
    float RT_CURV_CURVATURE;
    float RT_CURV_ZOOM;
    float RT_CURV_TILT_ANGLE_H;
    float RT_CURV_TILT_ANGLE_V;
    float RT_CURV_F;
    float RT_CURV_ROUNDED_CORNER;
    float RT_CURV_ANISOTROPIC_LVL;
    float RT_CURV_AA_STRENGTH;
}
param;

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; }
global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec3 prim_ray_o;
layout(location = 1) out vec3 prim_ray_d;
layout(location = 2) out vec3 plane_n;
layout(location = 3) out vec3 plane_u;
layout(location = 4) out vec3 plane_v;
layout(location = 5) out vec2 input_aspect;

void main() {
    gl_Position = global.MVP * Position;

    // Define curved object: sphere or cylinder.
    // Simplification: Assume origin of curved object (sphere / cylinder) is at
    // the origin.
    const float r =
        param.RT_CURV_CURVATURE > 1.0e-3 ? 1.0 / param.RT_CURV_CURVATURE : 1.0;
    // TODO: Why does this need an extra flip compared to the usage in
    // input_transform.inc?
    const vec2 tilt = get_rotated_vector(vec2(param.RT_CURV_TILT_ANGLE_H,
                                              param.RT_CURV_TILT_ANGLE_V),
                                         param.Rotation) *
                      (1 - 2 * (int(param.Rotation) % 2));
    const vec2 sin_tilt = sin(tilt);
    const vec2 cos_tilt = cos(tilt);
    // Simplification: Assume cylinder axis == plane_v.

    // Define input texture plane.
    // Basic vectors are:
    // n = (0, 0, -r)
    // u = (1, 0, 0)
    // v = (0, 1, 0)
    // n serves a double duty as the origin of the plane.
    // These are then rotated around x by tilt.x and then around y by tilt.y.
    plane_n = vec3(r * sin_tilt.x * cos_tilt.y, -r * sin_tilt.y,
                   -r * cos_tilt.x * cos_tilt.y);
    plane_u = vec3(cos_tilt.x, 0.0, sin_tilt.x);
    plane_v =
        vec3(sin_tilt.x * sin_tilt.y, cos_tilt.y, -cos_tilt.x * sin_tilt.y);

    // Compute aspects ratios.
    // Output aspect ratio does not need to be rotated since RA takes care of
    // output rotation.
    input_aspect =
        param.RT_CURV_ASPECT_H > 0.0 && param.RT_CURV_ASPECT_V > 0.0
            ? get_rotated_size(
                  vec2(param.RT_CURV_ASPECT_H, param.RT_CURV_ASPECT_V) /
                      max(param.RT_CURV_ASPECT_H, param.RT_CURV_ASPECT_V),
                  param.Rotation)
            : param.OriginalSize.xy /
                  max(param.OriginalSize.x, param.OriginalSize.y);
    const vec2 output_aspect =
        param.OutputSize.xy / max(param.OutputSize.x, param.OutputSize.y);

    // Simplification: Assume plane_o = plane_n
    generate_ray(TexCoord, input_aspect, output_aspect, plane_n, plane_u,
                 plane_v, param.RT_CURV_CURVATURE, param.RT_CURV_F,
                 param.RT_CURV_SHAPE, param.RT_CURV_ZOOM, prim_ray_o,
                 prim_ray_d);
}

#pragma stage fragment
layout(location = 0) in vec3 prim_ray_o;
layout(location = 1) in vec3 prim_ray_d;
layout(location = 2) in vec3 plane_n;
layout(location = 3) in vec3 plane_u;
layout(location = 4) in vec3 plane_v;
layout(location = 5) in vec2 input_aspect;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;

vec4 anisotropic_filtering(sampler2D tex, vec2 uv, float filtering_level,
                           vec3 rng_seed) {
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    mat2 ellipse_approx = 2.0 * param.RT_CURV_AA_STRENGTH * mat2(dx, dy);

    int num_samples = int(exp2(filtering_level));

    float max_axis_sq = max(dot(dx, dx), dot(dy, dy));
    float max_axis = sqrt(max_axis_sq);

    float mip_level = max(0.5 * (log2(max_axis_sq * param.RT_CURV_AA_STRENGTH *
                                      param.RT_CURV_AA_STRENGTH) -
                                 filtering_level),
                          0.0);
    float mip_level_i;
    float mip_level_f = modf(mip_level, mip_level_i);

    vec4 color = vec4(0.0);
    float total_weight = 0.0;

    // Generate a random angle for this fragment
    // hash function from: https://www.shadertoy.com/view/4djSRW
    rng_seed = fract(rng_seed * .1031);
    rng_seed += dot(rng_seed, rng_seed.zyx + 31.32);
    float random_angle =
        fract((rng_seed.x + rng_seed.y) * rng_seed.z) * 6.28318530718;  // 2*pi
    mat2 rot = mat2(cos(random_angle), -sin(random_angle), sin(random_angle),
                    cos(random_angle));

    mat2 tf = ellipse_approx * rot;

    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    const vec2 alpha = vec2(0.7548776662, 0.56984029);

    for (int i = 0; i < num_samples; i++) {
        // Generate rotated R2 sequence sample
        vec2 offset = fract(float(i) * alpha) - 0.5;

        // Calculate weight
        // Simplified cubic pulse (assuming center=0, w=sqrt(2)/2)
        float t = length(offset) * 1.41421;
        float weight = 1.0 - t * t * (3.0 - 2.0 * t);

        // Rotate and distort to ellipse
        offset = tf * offset;

        // Implement blending between mip-map levels in software because current
        // implementation is broken. See:
        // https://github.com/libretro/RetroArch/issues/16567;
        color +=
            mix(textureLod(tex, uv + offset, mip_level_i),
                textureLod(tex, uv + offset, mip_level_i + 1.0), mip_level_f) *
            weight;
        total_weight += weight;
    }

    return color / total_weight;
}

void main() {
    vec2 uv = trace_ray(input_aspect, prim_ray_o, prim_ray_d, plane_n, plane_u,
                        plane_v, param.RT_CURV_CURVATURE, param.RT_CURV_SHAPE);

    FragColor = anisotropic_filtering(
        Source, uv, param.RT_CURV_ANISOTROPIC_LVL,
        vec3(0.0));

    // Apply rounded corner darkening.
    const vec2 q =
        input_aspect * (abs(uv - 0.5) - 0.5) + param.RT_CURV_ROUNDED_CORNER;
    const float rounded_rect_dist = min(max(q.x, q.y), 0.0) +
                                    length(max(q, 0.0)) -
                                    param.RT_CURV_ROUNDED_CORNER;
    const float corner_darkening = smoothstep(
        -max(param.OutputSize.z, param.OutputSize.w), 0.0, -rounded_rect_dist);
    FragColor.rgb *= corner_darkening;
}
