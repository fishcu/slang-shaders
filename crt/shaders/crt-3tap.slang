#version 450

/*
    crt-3tap v1.0 by fishku
    Copyright (C) 2023
    Public domain license (CC0)

    Changelog:
    v1.0: Initial release.
*/

// clang-format off
#pragma parameter CRT1TAP_SETTINGS "=== CRT-3tap v1.0 settings ===" 0.0 0.0 1.0 1.0
#pragma parameter MIN_THICK "MIN_THICK: Scanline thickness of dark pixels." 0.3 0.0 1.4 0.05
#pragma parameter MAX_THICK "MAX_THICK: Scanline thickness of bright pixels." 0.8 0.0 1.4 0.05
#pragma parameter SHARP "SHARP: Sharpness of the scanline boundary." 1.0 0.0 1.0 0.05
#pragma parameter SUBPX_POS "SUBPX_POS: Scanline subpixel position." 0.0 -0.5 0.5 0.01
#pragma parameter THICK_FALLOFF "THICK_FALLOFF: Reduction of thinner scanlines." 0.65 0.2 2.0 0.05
// clang-format on

layout(push_constant) uniform Push {
    vec4 OutputSize;
    vec4 SourceSize;
    float MIN_THICK;
    float MAX_THICK;
    float SHARP;
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

// Fraction of radius at which area of segment equals area of other part.
// #define AREA_EQUIB 0.40397275
#define AREA_EQUIB (1 - 0.40397275)
// #define AREA_EQUIB 0.40397275 * 2.0

// Linear function that crosses 0.5 from above at the given offset with the given slope.
// Clamped to [0, 1]
float soft_mix_weight(float x, float offset, float slope) {
    return clamp(0.5 - slope * (x - offset), 0.0, 1.0);
}

vec3 color_to_radii(vec3 c) {
    return pow(mix(param.MIN_THICK.xxx, param.MAX_THICK.xxx, c), param.THICK_FALLOFF.xxx) * 0.5f;
}

void draw_scanline(inout float existing, float c, float coord_y_fract, float radius, float w) {
    existing =
        mix(existing, c, w * soft_mix_weight(coord_y_fract * coord_y_fract, radius * radius, 10.0));
}

void draw_circle(inout float existing, float c, vec2 coord, vec2 center, float radius) {
    const vec2 delta = coord - center;
    existing = mix(existing, c, soft_mix_weight(dot(delta, delta), radius * radius, 10.0));
}

void swap(inout float a, inout float b) {
    const float tmp = a;
    a = b;
    b = tmp;
}

void main() {
    vec3 c[3] = { texture(Source, vTexCoord - vec2(param.SourceSize.z, 0.0)).rgb,
                  texture(Source, vTexCoord).rgb,
                  texture(Source, vTexCoord + vec2(param.SourceSize.z, 0.0)).rgb };

    const vec2 coord_src_px = vTexCoord * param.SourceSize.xy - vec2(0.0, param.SUBPX_POS);
    vec2 coord_src_int;
    const vec2 coord_src_fract = modf(coord_src_px, coord_src_int) - vec2(0, 0.5);

    FragColor = vec4(vec3(0.0), 1.0);

    vec3 radii[3] = { color_to_radii(c[0]), color_to_radii(c[1]), color_to_radii(c[2]) };
    // const vec3 spot_center[3] = {- radii[0] * AREA_EQUIB, }
    vec3 scanline_weight[3] = {
        vec3(1.0 - coord_src_fract.x, 1.0 - coord_src_fract.x, 1.0 - coord_src_fract.x),
        vec3(1.0, 1.0, 1.0), vec3(coord_src_fract.x, coord_src_fract.x, coord_src_fract.x)
    };

    // for (int i = 0; i < 3; ++i) {
    //     if (c[0][i] > c[1][i]) {
    //         swap(c[0][i], c[1][i]);
    //         swap(radii[0][i], radii[1][i]);
    //         swap(scanline_weight[0][i], scanline_weight[1][i]);
    //     }
    //     if (c[0][i] > c[2][i]) {
    //         swap(c[0][i], c[2][i]);
    //         swap(radii[0][i], radii[2][i]);
    //         swap(scanline_weight[0][i], scanline_weight[2][i]);
    //     }
    //     if (c[1][i] > c[2][i]) {
    //         swap(c[1][i], c[2][i]);
    //         swap(radii[1][i], radii[2][i]);
    //         swap(scanline_weight[1][i], scanline_weight[2][i]);
    //     }
    // }

    // Draw from smallest to biggest
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         draw_scanline(FragColor[j], c[i][j], coord_src_fract.y, radii[i][j],
    //                       scanline_weight[i][j]);
    //     }
    // }

#if 0
    const vec3 radii_l = color_to_radii(l);
    const vec3 radii_m = color_to_radii(m);
    const vec3 radii_r = color_to_radii(r);
    const vec3 spot_center_l = radii_l * AREA_EQUIB;
    const vec3 spot_center_m = radii_m * AREA_EQUIB;
    const vec3 spot_center_r = radii_r * AREA_EQUIB;
    for (int c = 0; c < 3; ++c) {
        // Draw base layer.
        // If the neighboring pixel is equal or bigger, the center scanline is drawn up to the
        // pixel edge.
        // If the neighboring pixel is smaller, the center scanline is drawn up to the circle
        // beginning, and the circle is drawn.
        if (coord_src_fract.x < spot_center_m[c] * float(m[c] > l[c])) {
            draw_scanline(FragColor[c], l[c], coord_src_fract.y, radii_l[c]);
            draw_circle(FragColor[c], m[c], coord_src_fract, vec2(spot_center_m[c], 0.0),
                        radii_m[c]);
        } else if (coord_src_fract.x > 1 - spot_center_m[c] * float(m[c] > r[c])) {
            draw_scanline(FragColor[c], r[c], coord_src_fract.y, radii_r[c]);
            draw_circle(FragColor[c], m[c], coord_src_fract, vec2(1.0 - spot_center_m[c], 0.0),
                        radii_m[c]);
        } else {
            // For the area in-between, just draw the scanline.
            draw_scanline(FragColor[c], m[c], coord_src_fract.y, radii_m[c]);
        }
        // Draw the neighboring circles on top if their color is bigger.
        if (l[c] > m[c]) {
            draw_circle(FragColor[c], l[c], coord_src_fract, vec2(-spot_center_l[c], 0.0),
                        radii_l[c]);
        }
        if (r[c] > m[c]) {
            draw_circle(FragColor[c], r[c], coord_src_fract, vec2(1.0 + spot_center_r[c], 0.0),
                        radii_r[c]);
        }
    }
#endif
}
