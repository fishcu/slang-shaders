#version 450

/*
    crt-2tap v1.0 by fishku
    Copyright (C) 2023
    Public domain license (CC0)

    Changelog:
    v1.0: Initial release.
*/

// clang-format off
#pragma parameter CRT2TAP_SETTINGS "=== CRT-2tap v1.0 settings ===" 0.0 0.0 1.0 1.0
#pragma parameter MIN_THICK "Scanline thickness of dark pixels" 0.3 0.0 1.4 0.05
#pragma parameter MAX_THICK "Scanline thickness of bright pixels" 0.9 0.0 1.4 0.05
#pragma parameter V_SHARP "Vertical sharpness of the scanline" 1.0 0.0 10.0 0.1
#pragma parameter H_SHARP "Horizontal sharpness of pixel transitions" 2.0 2.0 20.0 0.5
#pragma parameter SUBPX_POS "Scanline subpixel position" 0.0 -0.5 0.5 0.01
#pragma parameter THICK_FALLOFF "Reduction / increase of thinner scanlines" 0.65 0.2 2.0 0.01
// clang-format on

layout(push_constant) uniform Push {
    vec4 OutputSize;
    vec4 SourceSize;
    float MIN_THICK;
    float MAX_THICK;
    float V_SHARP;
    float H_SHARP;
    float SUBPX_POS;
    float THICK_FALLOFF;
}
param;

layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 MVP;
}
global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;

void main() {
    gl_Position = global.MVP * Position;
    vTexCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 1) uniform sampler2D Source;
layout(set = 0, binding = 2) uniform sampler2D Original;

float color_to_radius(float signal) {
    return pow(mix(param.MIN_THICK, param.MAX_THICK, signal), param.THICK_FALLOFF) * 0.5;
}

float line_dist(vec2 coord, vec2 center, bool left) {
    if (left && coord.x > center.x || !left && coord.x < center.x) {
        const vec2 delta = coord - center;
        return sqrt(dot(delta, delta));
    }
    return abs(center.y - coord.y);
}

float extruded_stadium_dist(vec2 coord, vec2 center, float radius, float height, bool left) {
    const float dz = 0.5 * (1.0 - height) / param.H_SHARP;
    float dist = 0.0;
    if (left && coord.x > center.x || !left && coord.x < center.x) {
        const vec2 delta = coord - center;
        dist = sqrt(dot(delta, delta));
    } else {
        dist = abs(center.y - coord.y);
    }
    dist = max(0.0, dist - radius);
    return sqrt(dist * dist + dz * dz);
}

float smooth_min(float a, float b, float p) {
    // const float p = -1.0;
    return pow(pow(abs(a), p) + pow(abs(b), p), 1.0 / p);

    // case for p = -1.0:
    // return 1.0 / (1.0 / a + 1.0 / b);
}

float smooth_max(float a, float b) {
    const float p = param.V_SHARP;
    return pow(pow(abs(a), p) + pow(abs(b), p), 1.0 / p);

    // case for p = -1.0:
    // return 1.0 / (1.0 / a + 1.0 / b);
}

float screen_blend(float a, float b) {
    float result = 1.0f - (1.0f - a) * (1.0f - b);
    return result;
}

vec2 smin3(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    float m = h * h * h * 0.5;
    float s = m * k * (1.0 / 3.0);
    return (a < b) ? vec2(a - s, m) : vec2(b - s, 1.0 - m);
}

vec2 sminN(float a, float b, float k, float n) {
    float h = max(k - abs(a - b), 0.0) / k;
    float m = pow(h, n) * 0.5;
    float s = m * k / n;
    return (a < b) ? vec2(a - s, m) : vec2(b - s, 1.0 - m);
}

float smax3(float a, float b, float k) {
    k *= 1.4;
    float h = max(k - abs(a - b), 0.0);
    return max(a, b) + h * h * h / (6.0 * k * k);
}

vec2 smaxN(float a, float b, float k, float n) {
    float h = max(k - abs(a - b), 0.0) / k;
    float m = pow(h, n) * 0.5;
    float s = m * k / n;
    return (a > b) ? vec2(a + s, m) : vec2(b + s, 1.0 - m);
}

float expStep(float x, float k, float n) {
    k = pow(0.1, -n) * 0.693;
    return exp(-k * pow(x, n));
}

void blend_circles(vec2 coord,                 //
                   vec2 center1, vec3 color1,  //
                   vec2 center2, vec3 color2,  //
                   inout vec3 result) {
    for (int c = 0; c < 3; ++c) {
        const float radius1 = 0.7 * color_to_radius(color1[c]);
        const float radius2 = 0.7 * color_to_radius(color2[c]);
        center2 = vec2(center1.x + sqrt(max(0, radius1 * radius1 - radius2 * radius2)),  //
                       0.5);

        // EXTRUDED STADIUM DIST APPROACHES
        // ==========================================

        // const float dist1 = extruded_stadium_dist(coord, center1, radius1, color1[c], true);
        // float dist2 = extruded_stadium_dist(coord, center2, radius2, color2[c], false);
        // const vec2 res = smin3(dist1, dist2, param.V_SHARP * 0.07);
        // const float w = smoothstep(0.0, 1.0, 1.0 - res.x * param.H_SHARP);
        // result[c] = w; // * mix(color1[c], color2[c], res.y * res.y);

        // LINE DIST APPROACHES
        // ==========================================

        const float dist1 = line_dist(coord, center1, true) - radius1;
        const float dist2 = line_dist(coord, center2, false) - radius2;

        // Using my own smooth max
        // const float w1 = smoothstep(0.0, 1.0, 1.0 - dist1 * 20.0);
        // const float w2 = smoothstep(0.0, 1.0, 1.0 - dist2 * 20.0);
        // result[c] =
        //     clamp(smooth_max(w1 * color1[c], w2 * color2[c]), 0.0, max(color1[c], color2[c]));

        // Using sminN distances and blend weight
        // const float res = smooth_min(max(0.0, dist1), max(0.0, dist2), -param.V_SHARP);
        // const vec2 res = smin3(dist1 - color1[c]*0.1, dist2 - color2[c]*0.1, param.V_SHARP *
        // 0.07);
        // const float w = smoothstep(0.0, 1.0, 1.0 - res.x * 20.0);
        // result[c] = w * color1[c];
        // result[c] = w * mix(color1[c], color2[c], res.y);
        // result[c] = float(res <= 0.0);

        // Using smax(N)
        const float w1 = smoothstep(0.0, 1.0, (0.5 - dist1 * param.H_SHARP));
        const float w2 = smoothstep(0.0, 1.0, (0.5 - dist2 * param.H_SHARP));
        // const float w1 = expStep(max(0.0, dist1 + 0.1), 1.0, param.H_SHARP);
        // const float w2 = expStep(max(0.0, dist2 + 0.1), 1.0, param.H_SHARP);
        const float res = smooth_max(w1 * color1[c], w2 * color2[c]);
        // const float dist = smin3(dist1, dist2, param.V_SHARP * 0.07).x;
        // const float w = smoothstep(0.0, 1.0, 1.0 - dist * 20.0);
        result[c] = res;
        // result[c] = clamp(res, 0.0, max(color1[c], color2[c]));

        // screen blend
        // const float w1 = smoothstep(0.0, 1.0, 1.0 - dist1 * 20.0);
        // const float w2 = smoothstep(0.0, 1.0, 1.0 - dist2 * 20.0);
        // const float res = screen_blend(w1 * color1[c], w2 * color2[c]);
        // result[c] = clamp(res, 0.0, max(color1[c], color2[c]));
    }
}

void main() {
    float src_x_int;
    const float src_x_fract = modf(vTexCoord.x * param.SourceSize.x, src_x_int);

    float src_y_int;
    const float src_y_fract =
        modf(vTexCoord.y * param.SourceSize.y - param.SUBPX_POS, src_y_int) - 0.5;

    const vec3 signal =
        texture(Source, (vec2(src_x_int, src_y_int) + 0.5) * param.SourceSize.zw).rgb;

    FragColor = vec4(0.0, 0.0, 0.0, 1.0);

    const vec2 center1 = vec2(0.5, 0.5);
    const vec3 color1 = vec3(0.7, 0.7, 0.7);
    const vec2 center2 = center1;
    const vec3 color2 = vec3(0.6, 0.6, 0.6);

    blend_circles(vTexCoord,        //
                  center1, color1,  //
                  center2, color2,  //
                  FragColor.rgb);
}
