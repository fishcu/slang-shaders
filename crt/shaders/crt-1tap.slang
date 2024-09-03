#version 450

/*
    crt-1tap v1.2 by fishku
    Copyright (C) 2023
    Public domain license (CC0)

    Extremely fast and lightweight dynamic scanline shader.
    Contrasty and sharp output. Easy to configure.
    Can be combined with other shaders.

    How it works: Uses a single texture "tap" per pixel, hence the name.
    Exploits bilinear interpolation plus local coordinate distortion to shape
    the scanline. A pseudo-sigmoid function with a configurable slope at the
    inflection point is used to control horizontal smoothness.
    Uses a clamped linear function to anti-alias the edge of the scanline while
    avoiding further branching. The final thickness is shaped with a gamma-like
    operation to control overall image contrast.
    The scanline subpixel position can be controlled to achieve the best
    sharpness and uniformity of the output given different input sizes, e.g.,
    for even and odd integer scaling.

    Changelog:
    v1.3: Revamp the algorithm for brighter output.
    v1.2: Better scanline sharpness; Minor cleanups.
    v1.1: Update license; Better defaults; Don't compute alpha.
    v1.0: Initial release.
*/

// clang-format off
#pragma parameter CRT1TAP_SETTINGS "=== CRT-1tap v1.3 settings ===" 0.0 0.0 1.0 1.0
#pragma parameter MIN_THICK "Scanline thickness of dark pixels" 0.1 0.0 1.0 0.05
#pragma parameter MAX_THICK "Scanline thickness of bright pixels" 1.0 0.0 1.0 0.05
#pragma parameter V_SHARP "Vertical sharpness of the scanline" 0.5 0.0 1.0 0.05
#pragma parameter H_SHARP "Horizontal sharpness of pixel transitions" 0.15 0.0 1.0 0.05
#pragma parameter SUBPX_POS "Scanline subpixel position" 0.3 -0.5 0.5 0.01
#pragma parameter THICK_FALLOFF "Reduction / increase of thinner scanlines" 0.65 0.2 2.0 0.05
// clang-format on

layout(push_constant) uniform Push {
    vec4 OriginalSize;
    vec4 OutputSize;
    float MIN_THICK;
    float MAX_THICK;
    float V_SHARP;
    float H_SHARP;
    float SUBPX_POS;
}
param;

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; }
global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 tx_coord;

void main() {
    gl_Position = global.MVP * Position;
    tx_coord = TexCoord * param.OriginalSize.xy - vec2(-0.5, param.SUBPX_POS);
}

#pragma stage fragment
layout(location = 0) in vec2 tx_coord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 1) uniform sampler2D Source;
layout(set = 0, binding = 2) uniform sampler2D Original;

float slopestep(float x, float slope) {
    float s = sign(x - 0.5);
    float o = (1.0 + s) * 0.5;
    return o - 0.5 * s * pow(2.0 * (o - s * x), slope);
}

// Cumulative energy function
vec3 cef(float x, float m, vec3 x0, vec3 y0) {
    const vec3 xs = clamp((x - x0) / m, vec3(0.0), vec3(1.0));
    const vec3 xs2 = xs * xs;
    return y0 * (min(x0, vec3(x)) + m * (xs - xs * xs2 + 0.5 * xs2 * xs2));
}

void main() {
    vec2 tx_coord_i;
    vec2 tx_coord_f = modf(tx_coord, tx_coord_i);

    tx_coord_f.x = slopestep(tx_coord_f.x, mix(1.0, 6.0, param.H_SHARP));

    vec3 signal = texture(Source, (tx_coord_i + vec2(tx_coord_f.x - 0.5, 0.5)) *
                                      param.OriginalSize.zw)
                      .rgb;
    signal *= signal * param.MAX_THICK;

    // Variables that govern the scanline shape
    const float m = param.OriginalSize.y * param.OutputSize.w * 3.0;
    const float min_thick = param.MIN_THICK * 0.5;
    const vec3 x0 = max(vec3(min_thick), 0.5 * (signal - m));
    const vec3 y0 = min(vec3(1.0), signal / (2.0 * min_thick + m));

    tx_coord_f.y -= 0.5;
    float upper_edge =
        tx_coord_f.y - 0.5 * param.OutputSize.w * param.OriginalSize.y;
    float lower_edge =
        tx_coord_f.y + 0.5 * param.OutputSize.w * param.OriginalSize.y;
    if (upper_edge * lower_edge > 0.0) {
        // both on same side of half
        // cef is monotonic, so can always take the absolute difference.
        FragColor.rgb = abs(cef(abs(upper_edge), m, x0, y0) -
                            cef(abs(lower_edge), m, x0, y0));
    } else {
        // opposite sides -- treat as two pixels with other border at center
        FragColor.rgb = cef(abs(upper_edge), m, x0, y0) +
                               cef(abs(lower_edge), m, x0, y0);
    }
    FragColor.rgb = sqrt(FragColor.rgb * param.OriginalSize.w * param.OutputSize.y);
    FragColor.a = 1.0;
}
