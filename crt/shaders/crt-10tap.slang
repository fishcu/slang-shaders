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
#pragma parameter MIN_THICK "Scanline thickness of dark pixels" 0.1 0.0 1.4 0.05
#pragma parameter MAX_THICK "Scanline thickness of bright pixels" 0.9 0.0 1.4 0.05
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

// float to_lin(float x) {
//     return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
// }

// vec3 to_lin(vec3 x) { return vec3(to_lin(x.r), to_lin(x.g), to_lin(x.b)); }

// float to_srgb(float x) {
//     return x <= 0.0031308 ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
// }

// vec3 to_srgb(vec3 x) { return vec3(to_srgb(x.r), to_srgb(x.g), to_srgb(x.b));
// }

vec3 get_dist_signal(float y_fract, vec3 signal) {
    // Distributes energy from the scanline gap to the scanline
    // I.e., maximally saturated values are used on the scanline,
    // and black is used for the scanline gap.
    // Intermediate values are used for the purpose of anti-aliasing.
    // const float eff_aa_strength = sqrt(param.AA_STRENGTH);
    const float m =
        param.OriginalSize.w * param.OutputSize.y / param.AA_STRENGTH;
    const float x = abs(y_fract - 0.5);

    // Handle both cases, first element of mix is c when no part of scanline is
    // fully saturated, second element of mix is when at least some part of the
    // scanline is fully saturated.
    const vec3 c = mix(sqrt(m * signal), 0.5 + m * signal * 0.5,
                       step(vec3(1.0 / m), signal));
    return clamp(c - m * x, vec3(0.0), vec3(1.0));
}

vec3 cumulative_energy_function(float x, float m, vec3 x0, vec3 y0) {
    vec3 result;
    for (int i = 0; i < 3; ++i) {
        if (x <= x0[i]) {
            result[i] = x * y0[i];
        } else if (x < x0[i] + m) {
            const float xs = (x - x0[i]) / m;
            const float xs2 = xs * xs;
            result[i] = x0[i] * y0[i] + xs * xs2 - 0.5 * xs2 * xs2;
        } else {
            result[i] = x0[i] * y0[i] + 0.5 * m * y0[i];
        }
    }
    return result;
}

void main() {
    // FragColor.rgb = to_lin(texture(Original, vTexCoord).rgb);
    // FragColor[3] = 1.0;
    // return;

    // vec2 src_tx_int;
    // const vec2 src_tx_fract =
    //     modf(vTexCoord * param.OriginalSize.xy, src_tx_int);

    // const vec2 offsets[] = get_offsets(src_tx_fract);
    // const vec2 src_sample_coords[] = {
    //     src_tx_int + offsets[0], src_tx_int + offsets[1],
    //     src_tx_int + offsets[2], src_tx_int + offsets[3],
    //     src_tx_int + offsets[4], src_tx_int + offsets[5],
    //     src_tx_int + offsets[6], src_tx_int + offsets[7],
    //     src_tx_int + offsets[8], src_tx_int + offsets[9]};

    // FragColor = texture(Original, (src_tx_int + vec2(0.5 +
    // smoothstep(0.0, 1.0, src_tx_fract.x), 0.5)) * param.OriginalSize.zw);

    const vec3 signal =
        to_lin(texture(Source, vTexCoord).rgb) * param.MAX_THICK;

    // Get the brightest possible value within the current pixel, which is the
    // one closest to the scanline.
    // const float nearest_y =
    //     clamp((floor(vTexCoord.y * param.OriginalSize.y) + 0.5) *
    //               param.OriginalSize.w,
    //           vTexCoord.y - param.OutputSize.w * 0.5,
    //           vTexCoord.y + param.OutputSize.w * 0.5);
    // const float nearest_y_fract = fract(nearest_y * param.OriginalSize.y);
    // FragColor.rgb = get_dist_signal(nearest_y_fract, signal);

    // FragColor.rgb =
    //     get_dist_signal(fract(vTexCoord.y * param.OriginalSize.y), signal);

    // const int msaa_samples = 15;
    // vec3 accum_signal = vec3(0.0);
    // float floor_y = floor(vTexCoord.y * param.OutputSize.y);
    // for (int i = 0; i < msaa_samples; ++i) {
    //     accum_signal +=
    //         get_dist_signal(fract((floor_y + float(i) / msaa_samples) *
    //                               param.OutputSize.w * param.OriginalSize.y),
    //                         signal);
    // }
    // FragColor.rgb = accum_signal / msaa_samples;

    // const float tx_per_px = param.OriginalSize.w * param.OutputSize.y;
    // const float px_upper =
    //     (vTexCoord.y - param.OutputSize.w * 0.5) * param.OriginalSize.y;
    // const float px_lower = px_upper + param.OutputSize.w *
    // param.OriginalSize.y; const float scanline_center =
    //     floor(vTexCoord.y * param.OriginalSize.y) + 0.5;
    // if (scanline_center > px_upper && scanline_center < px_lower) {
    //     // Current pixel encompasses the scanline center. Handle each "side"
    //     of
    //     // the scanline.
    //     FragColor.rgb =
    //         tx_per_px *
    //         (clamp(signal * 0.5, vec3(0.0), vec3(scanline_center - px_upper))
    //         +
    //          clamp(signal * 0.5, vec3(0.0), vec3(px_lower -
    //          scanline_center)));
    // } else {
    //     // nearest_y is the pixel border facing towards the scanline center.
    //     const float nearest_y = clamp(scanline_center, px_upper, px_lower);
    //     const vec3 energy_left_over =
    //         signal * 0.5 - abs(fract(nearest_y) - 0.5);
    //     FragColor.rgb =
    //         clamp(energy_left_over * tx_per_px, vec3(0.0), vec3(1.0));
    // }

    const float min_thick = param.MIN_THICK * 0.5;
    const float m =
        param.OriginalSize.y * param.OutputSize.w * param.AA_STRENGTH;
    const float px_upper =
        (vTexCoord.y - param.OutputSize.w * 0.5) * param.OriginalSize.y;
    const float px_lower = px_upper + param.OutputSize.w * param.OriginalSize.y;
    const float scanline_center =
        floor(vTexCoord.y * param.OriginalSize.y) + 0.5;
    // decision boundary
    const vec3 signal_step = step(vec3(2.0 * min_thick + 0.5 * m), signal);
    const vec3 x0 = mix(vec3(min_thick), 0.5 * (signal - m), signal_step);
    const vec3 y0 =
        mix((sqrt(min_thick * min_thick + signal * m) - min_thick) / m,
            vec3(1.0), signal_step);
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

    FragColor.rgb = to_srgb(FragColor.rgb);
    FragColor[3] = 1.0;
}
