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

    const vec3 signal = to_lin(texture(Source, vTexCoord).rgb);

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

    const float energy_fract_per_px = param.OriginalSize.y * param.OutputSize.w;
    const float nearest_y =
        clamp((floor(vTexCoord.y * param.OriginalSize.y) + 0.5) *
                  param.OriginalSize.w,
              vTexCoord.y - param.OutputSize.w * 0.5,
              vTexCoord.y + param.OutputSize.w * 0.5);
    if (nearest_y > vTexCoord.y - param.OutputSize.w * 0.5 &&
        nearest_y < vTexCoord.y + param.OutputSize.w * 0.5) {
        // Current pixel encompasses the scanline center. Handle each "side" of
        // the scanline.
        // nearest_y is the scanline center.
        const float d1 = nearest_y - (vTexCoord.y - param.OutputSize.w * 0.5);
        const float d2 = (vTexCoord.y + param.OutputSize.w * 0.5) - nearest_y;
        FragColor.rgb = clamp(signal * 0.5 / energy_fract_per_px,
                                   vec3(0.0), vec3(d1 * param.OutputSize.y)) +
                        clamp(signal * 0.5 / energy_fract_per_px,
                                   vec3(0.0), vec3(d2 * param.OutputSize.y));
    } else {
        // nearest_y is the pixel border facing towards the scanline center.
        const float dist = abs(fract(nearest_y * param.OriginalSize.y) - 0.5);
        const float energy_fract_spent = dist;
        const vec3 energy_left_over = signal * 0.5 - energy_fract_spent;
        FragColor.rgb =
            clamp(energy_left_over / energy_fract_per_px, vec3(0.0), vec3(1.0));
    }

    FragColor.rgb = to_srgb(FragColor.rgb);
    FragColor[3] = 1.0;
}
