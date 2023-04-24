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
#pragma parameter V_SHARP "Vertical sharpness of the scanline" 0.5 0.0 1.0 0.05
#pragma parameter H_SHARP "Horizontal sharpness of pixel transitions" 0.15 0.0 1.0 0.05
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

float line_weight(vec2 coord, vec2 center, float radius, bool left) {
    if (left && coord.x > center.x) {
        return 0.0;
    } else if (!left && coord.x < center.x) {
        return 0.0;
    }
    const float r1 = radius;
    const float r2 = radius + 0.05;
    const float delta = center.y - coord.y;
    return clamp((r2 * r2 - delta * delta) / (r2 * r2 - r1 * r1), 0.0, 1.0);
}

float circle_weight(vec2 coord, vec2 center, float radius) {
    const float r1 = radius;
    const float r2 = r1 + 0.05;

    const vec2 delta = center - coord;
    return clamp((r2 * r2 - dot(delta, delta)) / (r2 * r2 - r1 * r1), 0.0, 1.0);
}

float line_dist(vec2 coord, vec2 center, bool left) {
    if (left && coord.x > center.x || !left && coord.x < center.x) {
        const vec2 delta = coord - center;
        return sqrt(dot(delta, delta));
    }
    return abs(center.y - coord.y);
}

float screen_blend(float a, float b) {
    float result = 1.0f - (1.0f - a) * (1.0f - b);
    return result;
}

float smooth_min(float a, float b) {
    const float p = -1.0;
    return pow(pow(abs(a), p) + pow(abs(b), p), 1.0 / p);
}

void blend_circles(vec2 coord,                                //
                   float radius1, vec2 center1, vec3 color1,  //
                   float radius2, vec2 center2, vec3 color2,  //
                   inout vec3 result) {
#if 0
    // const float w1 = line_weight(coord, center1, radius1, true);
    // const float w2 = circle_weight(coord, center1, radius1);
    // const float max12 = max(w1, w2);

    // const float w3 = line_weight(coord, center2, radius2, false);
    // const float w4 = circle_weight(coord, center2, radius2);
    // const float max34 = max(w3, w4);

    for (int c = 0; c < 3; ++c) {
        const float w1 = line_weight(coord, center1, radius1 * color1[c], true);
        const float w2 = circle_weight(coord, center1, radius1 * color1[c]);
        const float max12 = max(w1, w2);

        const float w3 = line_weight(coord, center2, radius2 * color2[c], false);
        const float w4 = circle_weight(coord, center2, radius2 * color2[c]);
        const float max34 = max(w3, w4);

        // if (max34 > 0.9) {
        result[c] = screen_blend(color1[c] * max12, color2[c] * max34);
        // } else {
        // result[c] = max(color1[c] * max12, color2[c] * max34);
        // }
    }
#else
    for (int c = 0; c < 3; ++c) {
        const float dist1 = line_dist(coord, center1, true) - radius1 * color1[c];
        const float dist2 = line_dist(coord, center2, false) - radius2 * color2[c];
        const float dist = smooth_min(max(0, dist1), max(0, dist2));
        // const float w = clamp(1.0 - max(0, dist) * 20.0, 0.0, 1.0);
        const float w = smoothstep(1.0, 0.0, dist * 20.0);
        result[c] = w;
    }
#endif
}

void main() {
    float src_x_int;
    const float src_x_fract = modf(vTexCoord.x * param.SourceSize.x, src_x_int);

    float src_y_int;
    const float src_y_fract =
        modf(vTexCoord.y * param.SourceSize.y - param.SUBPX_POS, src_y_int) - 0.5;

    const vec3 signal =
        texture(Source, (vec2(src_x_int, src_y_int) + 0.5) * param.SourceSize.zw).rgb;
    // FragColor.rgb = signal;

    FragColor = vec4(0.0, 0.0, 0.0, 1.0);

    const float radius1 = 0.35;
    const vec2 center1 = vec2(0.3, 0.5);
    const vec3 color1 = vec3(1.0, 0.0, 0.0);
    const float radius2 = param.THICK_FALLOFF;
    // const vec2 center2 = vec2(center1.x + sqrt(max(0, radius1 * radius1 - radius2 * radius2)), //
    //                           0.5);
    const vec2 center2 = center1;
    const vec3 color2 = vec3(0.0, 1.0, 0.0);

    blend_circles(vTexCoord,                 //
                  radius1, center1, color1,  //
                  radius2, center2, color2,  //
                  FragColor.rgb);
}
