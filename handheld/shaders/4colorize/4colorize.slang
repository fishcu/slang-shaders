#version 450

/*
    4colorize v1.0 by fishku
    Copyright (C) 2024
    Public domain license (CC0)

    Changelog:
    v1.0: Initial release.
*/

#include "parameters.inc"
// #include "shared.inc"

#include "../../../include/colorspace-tools.h"

// Similar to smoothstep, but has a configurable slope at x = 0.5.
// Original smoothstep has a slope of 1.5 at x = 0.5
#define INSTANTIATE_SLOPESTEP(T)                               \
    T slopestep(T edge0, T edge1, T x, float slope) {          \
        x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);    \
        const T s = sign(x - 0.5);                             \
        const T o = (1.0 + s) * 0.5;                           \
        return o - 0.5 * s * pow(2.0 * (o - s * x), T(slope)); \
    }
INSTANTIATE_SLOPESTEP(float)

// clang-format off
// 4x4 Bayer matrix
const float bayer_4x4[16] = float[](
     0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
    12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
     3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
    15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0
);

// 8x8 Bayer matrix
const float bayer_8x8[64] = float[](
     0.0/64.0, 32.0/64.0,  8.0/64.0, 40.0/64.0,  2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
    48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
    12.0/64.0, 44.0/64.0,  4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0,  6.0/64.0, 38.0/64.0,
    60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, 22.0/64.0,
     3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0,  1.0/64.0, 33.0/64.0,  9.0/64.0, 41.0/64.0,
    51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
    15.0/64.0, 47.0/64.0,  7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0,  5.0/64.0, 37.0/64.0,
    63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0, 21.0/64.0
);

// Shifted some rows
// const float bayer_8x8[64] = float[](
//      0.0/64.0, 32.0/64.0,  8.0/64.0, 40.0/64.0,  2.0/64.0, 34.0/64.0, 10.0/64.0, 42.0/64.0,
//     48.0/64.0, 16.0/64.0, 56.0/64.0, 24.0/64.0, 50.0/64.0, 18.0/64.0, 58.0/64.0, 26.0/64.0,
//     38.0/64.0, 12.0/64.0, 44.0/64.0,  4.0/64.0, 36.0/64.0, 14.0/64.0, 46.0/64.0,  6.0/64.0, //38.0/64.0,
//     22.0/64.0, 60.0/64.0, 28.0/64.0, 52.0/64.0, 20.0/64.0, 62.0/64.0, 30.0/64.0, 54.0/64.0, //22.0/64.0,
//      3.0/64.0, 35.0/64.0, 11.0/64.0, 43.0/64.0,  1.0/64.0, 33.0/64.0,  9.0/64.0, 41.0/64.0,
//     51.0/64.0, 19.0/64.0, 59.0/64.0, 27.0/64.0, 49.0/64.0, 17.0/64.0, 57.0/64.0, 25.0/64.0,
//     37.0/64.0, 15.0/64.0, 47.0/64.0,  7.0/64.0, 39.0/64.0, 13.0/64.0, 45.0/64.0,  5.0/64.0, //37.0/64.0,
//     21.0/64.0, 63.0/64.0, 31.0/64.0, 55.0/64.0, 23.0/64.0, 61.0/64.0, 29.0/64.0, 53.0/64.0 //21.0/64.0
// );
// clang-format on

float dither(float luma, float dither, float strength) {
    float offset_luma = luma + (dither - 0.5) * 0.25 * strength;
    return clamp(floor(offset_luma * 4.0) / 3.0, 0.0, 1.0);
}

float dither_r2(float luma, vec2 tx_coord, float strength) {
    // Magic numbers are explained here:
    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    // Numerically improved constants from here:
    // https://www.shadertoy.com/view/mts3zN
    const vec2 alpha = vec2(0.245122333753, 0.4301597090025);
    float z = fract(dot(alpha, tx_coord));
    z = 1.0 - 2.0 * abs(z - 0.5);
    return dither(luma, z, strength);
}

float dither_bayer_4x4(float luma, vec2 tx_coord, float strength) {
    int x = int(tx_coord.x) % 4;
    int y = int(tx_coord.y) % 4;
    int index = y * 4 + x;
    float bayer = bayer_4x4[index];
    return dither(luma, bayer, strength);
}

float dither_bayer_8x8(float luma, vec2 tx_coord, float strength, uint frame) {
    int x = int(tx_coord.x) % 8;
    int y = int(tx_coord.y) % 8;

#if 0
    // Rotate indices for lols
    // int x = int(tx_coord.x);
    // int y = int(tx_coord.y);
    // Shift center of rotation
    // x -= 3;
    // y -= 3;
    // Determine angle
    float angle = fract(float(frame) * 0.000);
    // const float alpha_1 = 0.81917251339;
    // const float alpha_2 = 0.6710436067;
    // const float alpha_3 = 0.5497004779;
    // float angle = fract(alpha_1 * tx_coord.x / 10000.0 +
    //                     alpha_2 * tx_coord.y / 10000.0 + alpha_3 * luma *
    //                     123.0);
    // angle = 1.0 - 2.0 * abs(angle - 0.5);
    // float angle = fract(luma / 2.0);
    // If outside range of -90 to 90 degrees, flip
    if (angle > 0.5) {
        x = -x;
        y = -y;
        angle -= 0.5;
    }
    // Convert to rads and shift by -90 degrees
    angle = (angle - 0.25) * 2.0 * 3.14159265;
    float t = -tan(angle / 2.0);
    float s = sin(angle);
    // Apply 3 shears for actual rotation
    x += int(round(t * y));
    y += int(round(s * x));
    x += int(round(t * y));
    // Shift center back
    // x += 3;
    // y += 3;
    // Normalize indices
    x = (x + 8 * 1000) % 8;
    y = (y + 8 * 1000) % 8;

    // return dither_r2(luma, vec2(float(x), float(y)), strength);
#endif

    int index = y * 8 + x;
    float bayer = bayer_8x8[index];
    return dither(luma, bayer, strength);
}

float dither_bayer_2x4x2(float luma, vec2 tx_coord, int frame, float strength) {
    int x = int(tx_coord.x) % 4;
    int y = int(tx_coord.y) % 2;
    int t = frame % 2;
    int index = x + y * 4 + t * 8;
    float bayer = bayer_4x4[index];
    return dither(luma, bayer, strength);
}

float dither_bayer_4x8x2(float luma, vec2 tx_coord, int frame, float strength) {
    int x = int(tx_coord.x) % 8;
    int y = int(tx_coord.y) % 4;
    int t = frame % 2;
    int index = x + y * 8 + t * 32;
    float bayer = bayer_8x8[index];
    return dither(luma, bayer, strength);
}

// float dither_r2_time(float luma, vec2 tx_coord, uint time, float strength) {
//     // Magic numbers are explained here:
//     //
//     https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
//     // const float alpha_1 = 0.81917251339;
//     // const float alpha_2 = 0.6710436067;
//     // const float alpha_3 = 0.5497004779;
//     const float alpha_1 = 0.7548776662;
//     const float alpha_2 = 0.56984029;
//     float t = float(time % 2) * 1.0;
//     float z = fract(alpha_1 * (tx_coord.x + t) + alpha_2 * (tx_coord.y + t));
//     z = 1.0 - 2.0 * abs(z - 0.5);
//     return dither(luma, z, strength);
// }

// https://www.shadertoy.com/view/llGcDm
int hilbert(ivec2 p, int level) {
    int d = 0;
    for (int k = 0; k < level; k++) {
        int n = level - k - 1;
        ivec2 r = (p >> n) & 1;
        d += ((3 * r.x) ^ r.y) << (2 * n);
        if (r.y == 0) {
            if (r.x == 1) {
                p = (1 << n) - 1 - p;
            }
            p = p.yx;
        }
    }
    return d;
}

// knuth's multiplicative hash function (fixed point R1)
uint kmhf(uint x) { return 0x80000000u + 2654435789u * x; }

// mapping each pixel to a hilbert curve index, then taking a value from the
// Roberts R1 quasirandom sequence for it
uint hilbert_r1_blue_noise(uvec2 p) {
    uint x = uint(hilbert(ivec2(p), 17)) % (1u << 17u);
    x = kmhf(x);
    return x;
}

// mapping each pixel to a hilbert curve index, then taking a value from the
// Roberts R1 quasirandom sequence for it
float hilbert_r1_blue_noisef(uvec2 p) {
    uint x = hilbert_r1_blue_noise(p);
    return float(x) / 4294967296.0;
}

vec3 triwave(vec3 z) {  // z in [0..1]
    return mix(2.0 * z, 2.0 - 2.0 * z, step(0.5, z));
}
float uniform_to_tri(float v) {
    v = v * 2.0 - 1.0;
    v = sign(v) * (1.0 - sqrt(1.0 - abs(v)));
    return v * 0.5 + 0.5;
}

float dither_hilbert_r1(float luma, vec2 tx_coord, float strength) {
    return dither(luma, hilbert_r1_blue_noisef(uvec2(tx_coord)), strength);
}

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
    uint FrameCount;
    float FCOL_CONT;
    float FCOL_DITHERING;
    float FCOL_DITHERING_STRENGTH;
}
param;

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; }
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
layout(set = 0, binding = 2) uniform sampler2D Input;
layout(set = 0, binding = 3) uniform sampler2D Source;

#define EDGES Source

void main() {
    // it's all in here man:
    // https://www.shadertoy.com/view/cl2GRm
#if 0
    // Debug

    // Apply square for fast "gamma correction".
    vec3 rgb = texture(Input, vTexCoord).rgb;
    float edges = texture(EDGES, vTexCoord, 1.0).r;
    rgb *= rgb;
    edges *= edges;

    // Luma conversion
    float res = luma(rgb);

    // res = edges;
    // res = res - 10.0 * edges;

    // Apply sqrt for fast "gamma correction".
    res = sqrt(res);

    // Apply contrast
    res = slopestep(0.0, 1.0, res, param.FCOL_CONT);

    FragColor = vec4(vec3(res), 1.0);

#else
    // Sample.
    // Apply square for fast "gamma correction".
    vec3 rgb = texture(Input, vTexCoord).rgb;
    rgb *= rgb;

    // Luma conversion
    float res = luma(rgb);

    // Apply sqrt for fast "gamma correction".
    res = sqrt(res);

    // Apply contrast
    res = slopestep(0.0, 1.0, res, param.FCOL_CONT);

    float edges = texture(EDGES, vTexCoord).r;
    float dither_strength =
        max(0.0, 1.0 - edges * 10.0) * param.FCOL_DITHERING_STRENGTH;
    // float dither_strength = 1.0 - step(0.05, edges);
    // float dither_strength = param.FCOL_DITHERING_STRENGTH;

    // res = vTexCoord.x;

    // Quantize to 4 colors.
    switch (int(param.FCOL_DITHERING)) {
        case 0:
        default:
            res = min(1.0, floor(4.0 * res) / 3.0);
            break;
        case 1:
            res = dither_bayer_4x4(res, vTexCoord * param.OutputSize.xy,
                                   dither_strength);
            break;
        case 2:
            res = dither_bayer_8x8(res, vTexCoord * param.OutputSize.xy,
                                   dither_strength, param.FrameCount);
            break;
        case 3:
            res = dither_bayer_2x4x2(res, vTexCoord * param.OutputSize.xy,
                                     int(param.FrameCount), dither_strength);
            break;
        case 4:
            res = dither_bayer_4x8x2(res, vTexCoord * param.OutputSize.xy,
                                     int(param.FrameCount), dither_strength);
            break;
        case 5:
            res = dither_r2(res, vTexCoord * param.OutputSize.xy,
                            dither_strength);
            break;
        case 6:
            const float e = 2.71828182845904523536028747;
            vec2 noiseOff = vec2(12345.0 * fract(float(param.FrameCount) * e));
            res = dither_hilbert_r1(res,
                                    vTexCoord * param.OutputSize.xy + noiseOff,
                                    dither_strength);
            break;
    };

    FragColor = vec4(vec3(res), 1.0);
#endif
}
