#version 450

/*
    Authentic GBC v1.0 by fishku
    Copyright (C) 2024
    Public domain license (CC0)

    Attempts to render GBC subpixels authentically.

    Reference photos:
    - https://gbcc.dev/technology/subpixels.jpg

    Inspired by:
    -
   https://www.reddit.com/r/AnaloguePocket/comments/1azaxgd/ive_made_some_improvements_to_my_analogue_pocket/

    Changelog:
    v1.0: Initial release.
*/

#include "parameters.inc"
#include "shared.inc"

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
    uint FrameCount;
    float AUTH_GBC_BRIG;
}
param;

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; }
global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec4 px_rect;
layout(location = 1) out vec2 tx_coord;
layout(location = 2) out vec2 tx_to_px;
layout(location = 3) out vec2 subpx_size;
layout(location = 4) out vec2 notch_size;
layout(location = 5) out float subpx_orig_y;
layout(location = 6) out vec2 vTexCoord;

void main() {
    gl_Position = global.MVP * Position;

    auth_gbc_vert_shader(param.SourceSize.xy, param.OutputSize.xy, param.AUTH_GBC_BRIG, TexCoord,
                         px_rect, tx_coord, tx_to_px, subpx_size, notch_size, subpx_orig_y);
    vTexCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec4 px_rect;
layout(location = 1) in vec2 tx_coord;
layout(location = 2) in vec2 tx_to_px;
layout(location = 3) in vec2 subpx_size;
layout(location = 4) in vec2 notch_size;
layout(location = 5) in float subpx_orig_y;
layout(location = 6) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

vec2 smooth_hash22(vec2 p) {
    vec2 s = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(s) * 43758.5453);
}

vec2 simplex_noise22(vec2 p) {
    const float K1 = 0.366025404;  // (sqrt(3)-1)/2
    const float K2 = 0.211324865;  // (3-sqrt(3))/6
    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    float m = step(a.y, a.x);
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n =
        h * h * h * h * vec3(dot(a, hash22(i)), dot(b, hash22(i + o)), dot(c, hash22(i + 1.0)));
    return vec2(dot(n, vec3(70.0)), dot(n, vec3(70.0, 70.0, 71.0)));
}

vec2 value_noise22(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // Smoothstep
    vec2 a = hash22(i);
    vec2 b = hash22(i + vec2(1.0, 0.0));
    vec2 c = hash22(i + vec2(0.0, 1.0));
    vec2 d = hash22(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

vec2 gaussian_rand(vec2 seed, vec2 mean, vec2 std_dev) {
    // Constants for Winitzki's approximation
    const float INV_ALPHA = 7.14285714;  // Precomputed 1.0 / 0.14
    const float K = 4.54717381;          // Precomputed 2.0 / (PI * 0.14)

    // Inline hash22 function
    vec2 x = simplex_noise22(seed);
    x = x * 2.0 - 1.0;

    // Inverse error function approximation for both components
    vec2 y = log(max(vec2(0.001), 1.0 - x * x)) * 0.5;
    vec2 z = vec2(K) + y;
    vec2 w = sqrt(max(vec2(0.0), z * z - y * INV_ALPHA));

    return sqrt(w - z) * sign(x) * std_dev + mean;
}

void main() {
#if 0
    // Debug
    vec3 res = pixel_color(px_rect, vec2(500.0), vec2(100.0));
    res += pixel_color(px_rect, vec2(500.0), vec2(600.0, 100.0));
    res += pixel_color(px_rect, vec2(500.0), vec2(100.0, 600.0));
    res += pixel_color(px_rect, vec2(500.0), vec2(600.0));
    FragColor = vec4(res, 1.0);
#else

    // Perturb tx_coord randomly to reduce inference patterns
    vec2 vTexCoord_p =
        vTexCoord + gaussian_rand(vTexCoord * vec2(60.0, 20.0), vec2(0.0), vec2(0.1, 0.01) * param.SourceSize.zw);
    vec4 px_rect_p =
        vec4(vTexCoord_p * param.OutputSize.xy - 0.5, vTexCoord_p * param.OutputSize.xy + 0.5);
    vec2 tx_coord_p = vTexCoord_p * param.SourceSize.xy;

    // Figure out 4 nearest texels in source texture
    vec2 tx_coord_i;
    const vec2 tx_coord_f = modf(tx_coord_p, tx_coord_i);
    const vec2 tx_coord_off = step(vec2(0.5), tx_coord_f) * 2.0 - 1.0;
    const vec2 tx_origins[] = {tx_coord_i, tx_coord_i + vec2(tx_coord_off.x, 0.0),
                               tx_coord_i + vec2(0.0, tx_coord_off.y), tx_coord_i + tx_coord_off};

    // Sample.
    // Apply square for fast "gamma correction".
    vec3 samples[] = {texture(Source, (tx_origins[0] + 0.5) * param.SourceSize.zw).rgb,
                      texture(Source, (tx_origins[1] + 0.5) * param.SourceSize.zw).rgb,
                      texture(Source, (tx_origins[2] + 0.5) * param.SourceSize.zw).rgb,
                      texture(Source, (tx_origins[3] + 0.5) * param.SourceSize.zw).rgb};
    samples[0] *= samples[0];
    samples[1] *= samples[1];
    samples[2] *= samples[2];
    samples[3] *= samples[3];

    // Apply shader.
    const vec3 res = samples[0] * pixel_color(px_rect_p, tx_to_px, tx_origins[0] * tx_to_px,
                                              subpx_orig_y, subpx_size, notch_size) +
                     samples[1] * pixel_color(px_rect_p, tx_to_px, tx_origins[1] * tx_to_px,
                                              subpx_orig_y, subpx_size, notch_size) +
                     samples[2] * pixel_color(px_rect_p, tx_to_px, tx_origins[2] * tx_to_px,
                                              subpx_orig_y, subpx_size, notch_size) +
                     samples[3] * pixel_color(px_rect_p, tx_to_px, tx_origins[3] * tx_to_px,
                                              subpx_orig_y, subpx_size, notch_size);

    // Apply sqrt for fast "gamma correction".
    FragColor = vec4(sqrt(res), 1.0);
#endif
}
