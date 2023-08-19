#version 450

/*
    crt-10tap v1.0 by fishku
    Copyright (C) 2023
    Public domain license (CC0)

    Changelog:
    v1.0: Initial release.
*/

// clang-format off
#pragma parameter CRT10TAP_SETTINGS "=== CRT-10tap v1.0 settings ===" 0.0 0.0 1.0 1.0
// #include "crt-Ntap-common-params.slang"
#pragma parameter MIN_THICK "Scanline thickness of dark pixels" 0.0 0.0 1.4 0.05
#pragma parameter MAX_THICK "Scanline thickness of bright pixels" 1.0 0.0 1.4 0.05
#pragma parameter S_GAMMA_OFFS "Signal gamma offset" 0.0 -2.0 2.0 0.1
#pragma parameter AA_STRENGTH "Smoothing strength" 1.0 0.5 4.0 0.1

// clang-format on

layout(push_constant) uniform Push {
    vec4 OriginalSize;
    vec4 SourceSize;
    vec4 OutputSize;
    float MIN_THICK;
    float MAX_THICK;
    float SUBPX_POS;
    float S_GAMMA_OFFS;
    float AA_STRENGTH;
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
layout(set = 0, binding = 1) uniform sampler2D Source;
layout(set = 0, binding = 2) uniform sampler2D Original;

// vec2[10] get_offsets(vec2 src_tx_fract) {
//     if (src_tx_fract.y < 0.5) {
//         return vec2[10](vec2(-2.0, -1.0), vec2(-1.0, -1.0), vec2(0.0, -1.0),
//                         vec2(1.0, -1.0), vec2(2.0, -1.0),  //
//                         vec2(-2.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 0.0),
//                         vec2(1.0, 0.0), vec2(2.0, 0.0));
//     } else {
//         return vec2[10](vec2(-2.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 0.0),
//                         vec2(1.0, 0.0), vec2(2.0, 0.0),  //
//                         vec2(-2.0, 1.0), vec2(-1.0, 1.0), vec2(0.0, 1.0),
//                         vec2(1.0, 1.0), vec2(2.0, 1.0));
//     }
// }

float[5] get_gaussian() {
    float result[5];
    // mean 0, std. dev. 1
    const float lower = -2.0;
    const float higher = 2.0;
    const float step = (higher - lower) / (5.0 - 1.0);
    float sum = 0.0;
    for (int i = 0; i < 5; ++i) {
        const float x = lower + step * i;
        result[i] = exp(-x * x / 2.0);
        sum += result[i];
    }
    for (int i = 0; i < 5; ++i) {
        result[i] /= sum;
    }
    return result;
}

vec3 to_lin(vec3 x) { return pow(x, vec3(2.2 + param.S_GAMMA_OFFS)); }

vec3 to_srgb(vec3 x) { return pow(x, vec3(1.0 / 2.2)); }

vec3 cumulative_energy_function(float x, float m, vec3 x0, vec3 y0) {
    const vec3 xs = clamp((x - x0) / m, vec3(0.0), vec3(1.0));
    const vec3 xs2 = xs * xs;
    return y0 * (min(x0, vec3(x)) + m * (xs - xs * xs2 + 0.5 * xs2 * xs2));
}

void main() {
    // FragColor.rgb = to_lin(texture(Original, vTexCoord).rgb);
    // FragColor[3] = 1.0;
    // return;

    // const vec2 offsets[] = get_offsets(src_tx_fract);
    // const vec2 src_sample_coords[] = {
    //     src_tx_int + offsets[0], src_tx_int + offsets[1],
    //     src_tx_int + offsets[2], src_tx_int + offsets[3],
    //     src_tx_int + offsets[4], src_tx_int + offsets[5],
    //     src_tx_int + offsets[6], src_tx_int + offsets[7],
    //     src_tx_int + offsets[8], src_tx_int + offsets[9]};

    const vec3 signal =
        to_lin(texture(Source, vTexCoord).rgb) * param.MAX_THICK;

    const float min_thick = param.MIN_THICK * 0.5;
    const float m =
        param.OriginalSize.y * param.OutputSize.w * param.AA_STRENGTH;
    const float px_upper =
        (vTexCoord.y - param.OutputSize.w * 0.5) * param.OriginalSize.y;
    const float px_lower = px_upper + param.OutputSize.w * param.OriginalSize.y;
    const float scanline_center =
        floor(vTexCoord.y * param.OriginalSize.y) + 0.5;

    const vec3 x0 = max(vec3(min_thick), 0.5 * (signal - m));
    const vec3 y0 = min(vec3(1.0), signal / (2.0 * min_thick + m));
    if (scanline_center > px_upper && scanline_center < px_lower) {
        // Current pixel encompasses the scanline center. Handle each "side" of
        // the scanline.
        FragColor.rgb =
            cumulative_energy_function(scanline_center - px_upper, m, x0, y0) +
            cumulative_energy_function(px_lower - scanline_center, m, x0, y0);
    } else {
        const float d1 = abs(fract(px_upper) - 0.5);
        const float d2 = abs(fract(px_lower) - 0.5);
        FragColor.rgb = abs(cumulative_energy_function(d1, m, x0, y0) -
                            cumulative_energy_function(d2, m, x0, y0));
    }
    FragColor.rgb *= param.OutputSize.y * param.OriginalSize.w;

    FragColor.rgb = to_srgb(FragColor.rgb);
    FragColor[3] = 1.0;

    // {
    //     const vec3 signal = vec3(0.04, 0.08, 0.12);
    //     // const vec3 signal = vec3(0.1, 0.3, 0.8);
    //     const vec3 x0 = max(vec3(min_thick), 0.5 * (signal - m));
    //     const vec3 y0 = min(signal / (2.0 * min_thick + m), vec3(1.0));

    //     const vec3 y = 10 * cumulative_energy_function(vTexCoord.x, m, x0,
    //     y0); const vec3 e = energy_function(vTexCoord.x, m, x0, y0); for (int
    //     i = 0; i < 3; ++i) {
    //         if (abs(vTexCoord.y - y[i]) < 0.01) {
    //             FragColor[i] = 1.0;
    //         }
    //     }

    //     if (abs(vTexCoord.y - 0.04 * 5) < 0.005) {
    //         FragColor = vec4(1.0);
    //     }
    //     if (abs(vTexCoord.y - 0.08 * 5) < 0.005) {
    //         FragColor = vec4(1.0);
    //     }
    //     if (abs(vTexCoord.y - 0.12 * 5) < 0.005) {
    //         FragColor = vec4(1.0);
    //     }
    // }
}
