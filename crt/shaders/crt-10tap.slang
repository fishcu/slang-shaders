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
#pragma parameter H_BLUR "Horizontal blur strength" 0.3 0.0 3.0 0.05

#pragma parameter H1 "Signal 1" 0.0 0.0 5.0 0.05
#pragma parameter H2 "Signal 2" 0.0 0.0 5.0 0.05
#pragma parameter ELLIPSE_W "Ellipse width" 1.0 0.0 3.0 0.05
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
    float H_BLUR;
    //
    float H1;
    float H2;
    float ELLIPSE_W;
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

float[5] get_gaussian() {
    float result[5];
    // mean 0, std. dev. 1
    const float lower = -1.5;
    const float higher = 1.5;
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

// Cumulative energy function
vec3 cef(float x, float m, vec3 x0, vec3 y0) {
    const vec3 xs = clamp((x - x0) / m, vec3(0.0), vec3(1.0));
    const vec3 xs2 = xs * xs;
    return y0 * (min(x0, vec3(x)) + m * (xs - xs * xs2 + 0.5 * xs2 * xs2));
}

vec3 sample_scanline(vec3 signal, float scanline_center, float px_upper,
                     float px_lower) {
    // Variables that govern the scanline shape
    const float m =
        param.OriginalSize.y * param.OutputSize.w * param.AA_STRENGTH;
    const float min_thick = param.MIN_THICK * 0.5;
    const vec3 x0 = max(vec3(min_thick), 0.5 * (signal - m));
    const vec3 y0 = min(vec3(1.0), signal / (2.0 * min_thick + m));

    if (scanline_center > px_upper && scanline_center < px_lower) {
        // Current pixel encompasses the scanline center. Handle each "side" of
        // the scanline.
        return cef(scanline_center - px_upper, m, x0, y0) +
               cef(px_lower - scanline_center, m, x0, y0);
    } else {
        // cef is monotonic, so can always take the absolute difference.
        return abs(cef(abs(px_upper - scanline_center), m, x0, y0) -
                   cef(abs(px_lower - scanline_center), m, x0, y0));
    }
}

#define PI 3.1415926538
#define EPS 1.0e-8

float f(vec2 x, vec2 center, vec2 axes) {
    const vec2 a = (x - center) / (axes + EPS);
    return dot(a, a) - 1.0;
}

vec2 gradf(vec2 x, vec2 center, vec2 axes) {
    // (x-center.x)^2 / a^2 + (y-center.y)^2 / b^2 - 1 = 0
    // df/dx = 2(x-center.x) / a^2;
    // df/dy = 2(y-center.y) / b^2;
    return 2.0 * (x - center) / (axes * axes + EPS);
}

float f2(vec2 x, vec2 center, vec2 axes) {
    // const float f = length((x - center) / axes);
    // return (f - 1.0) * f;
    const vec2 a = (x - center) / (axes + EPS);
    return length(a) - 1.0;
}

vec2 gradf2(vec2 x, vec2 center, vec2 axes) {
    // return (x - center) / axes / axes;
    // sqrt((x-center.x)^2 / a^2 + (y-center.y)^2 / b^2) - 1 = 0
    // df/dx = 1/sqrt(...) * 2(x-center.x) / a^2;
    // df/dy = 1/sqrt(...) * 2(y-center.y) / b^2;
    const vec2 a = (x - center) / (axes + EPS);
    return 2.0 * (x - center) / (length(a) * axes * axes + EPS);
}

vec3 sample_scanline_new(vec3 signal_1, vec3 signal_2, vec2 signal_center,
                         vec2 coord) {
    const float m =
        param.OriginalSize.y * param.OutputSize.w * param.AA_STRENGTH;
    const float min_thick = param.MIN_THICK * 0.5;

    const vec3 signal_smaller =
        mix(signal_1, signal_2, step(signal_2, signal_1));
    const vec3 signal_bigger =
        mix(signal_2, signal_1, step(signal_2, signal_1));

    const vec3 w1 = min(vec3(1.0), signal_smaller / (2.0 * min_thick + m));
    const vec3 w2 = min(vec3(1.0), signal_bigger / (2.0 * min_thick + m));

    const vec3 h1 = max(vec3(min_thick), 0.5 * (signal_smaller - m));
    const vec3 h2 = max(vec3(min_thick), 0.5 * (signal_bigger - m));

    // Ellipse scales with:
    // - user parameter
    // - height of scanline to avoid overly pointy ellipses
    // - aspect ratio of screen to invert stretching effect
    const vec3 eff_ellipse_w = param.ELLIPSE_W * h2 * param.OutputSize.z *
                               param.OutputSize.y * param.SourceSize.x *
                               param.SourceSize.w;

    // get center of ellipse
    const vec3 r = h1 / (h2 + EPS);
    const vec3 x = eff_ellipse_w *
                   (PI / 4.0 * h2 * (w2 - w1) +
                    0.5 * h2 * w1 * (acos(r) - r * sqrt(1 - r * r))) /
                   (w2 * h2 - w1 * h1 + EPS);
    const vec3 ellipse_center_x =
        signal_center.x + x * sign(signal_2 - signal_1);
    // const vec3 ellipse_center_x = vec3(signal_center.x);

    // Draw scanlines
    vec3 result;
    for (int i = 0; i < 3; ++i) {
        if (coord.x < ellipse_center_x[i]) {
            if (signal_1[i] < signal_2[i]) {
                result[i] =
                    w1[i] * (1 - smoothstep(h1[i] - 0.5 * m, h1[i] + 0.5 * m,
                                            abs(coord.y - signal_center.y)));
            } else {
                result[i] =
                    w2[i] * (1 - smoothstep(h2[i] - 0.5 * m, h2[i] + 0.5 * m,
                                            abs(coord.y - signal_center.y)));
            }
        } else {
            if (signal_1[i] < signal_2[i]) {
                result[i] =
                    w2[i] * (1 - smoothstep(h2[i] - 0.5 * m, h2[i] + 0.5 * m,
                                            abs(coord.y - signal_center.y)));
            } else {
                result[i] =
                    w1[i] * (1 - smoothstep(h1[i] - 0.5 * m, h1[i] + 0.5 * m,
                                            abs(coord.y - signal_center.y)));
            }
        }
    }

    // Draw ellipse
    for (int i = 0; i < 3; ++i) {
        const vec2 axes = vec2(eff_ellipse_w[i], h2[i]);
        const vec2 center = vec2(ellipse_center_x[i], signal_center.y);

        const float fx = f(coord, center, axes);
        const vec2 gradfx = gradf(coord, center, axes);
        const float dist = fx / (length(gradfx) + EPS);
        result[i] =
            max(result[i], w2[i] * (1.0 - smoothstep(-0.5 * m, 0.5 * m, dist)));
    }

    return result;
}

vec3 get_blended_value(vec3 samples[5], float x, float x_center) {
    vec3 signal = vec3(0.0);
    float g_sum = 0.0;
    for (float offset = -2.0; offset <= 2.0; offset += 1.0) {
        const float dx = (x + offset) - x_center;
        const float w =
            exp(-dx * dx / (param.H_BLUR * param.H_BLUR * 0.25 + EPS));
        g_sum += w;
        signal += w * samples[clamp(int(-x - offset + 3.0), 0, 4)];
    }
    // if (g_sum < 1.0e-6) {
    //     return samples[clamp(int(x + 2.0), 0, 4)];
    // }
    return signal / g_sum;
}

vec3 get_bilateral(sampler2D tex, vec2 coord, float x_sigma, float i_sigma) {
    x_sigma *= param.OriginalSize.z;
    vec3 center_i = to_lin(texture(tex, coord).rgb);

    if (x_sigma < EPS || i_sigma < EPS) {
        return center_i;
    }

    vec3 sum = vec3(0.0);
    vec3 result = vec3(0.0);

    vec3 right_last = center_i;
    vec3 left_last = center_i;
    vec3 right_updating = vec3(1.0);
    vec3 left_updating = vec3(1.0);
    for (float delta = 0.0; delta <= 127.0; delta += 1.0) {
        {
            float sample_x = coord.x + delta * param.OutputSize.z;
            float dx = sample_x - coord.x;
            vec3 w_x = exp(-vec3(dx * dx / (x_sigma * x_sigma + EPS)));

            vec3 s = to_lin(texture(tex, vec2(sample_x, coord.y)).rgb);

            vec3 di = right_last - s;

            vec3 eff_s;
            for (int i = 0; i < 3; ++i) {
                if (right_updating[i] < 0.5) {
                    eff_s[i] = right_last[i];
                    continue;
                }
                if (abs(di)[i] > i_sigma) {
                    right_updating[i] = 0.0;
                    eff_s[i] = min(s[i], right_last[i]);
                    right_last[i] = eff_s[i];
                    continue;
                }
                eff_s[i] = s[i];
                right_last[i] = s[i];
            }
            vec3 w_i = vec3(1.0);

            vec3 w = w_x * w_i;
            sum += w;
            result += w * eff_s;
        }
        if (delta > 0.0) {
            float sample_x = coord.x - delta * param.OutputSize.z;
            float dx = sample_x - coord.x;
            vec3 w_x = exp(-vec3(dx * dx / (x_sigma * x_sigma + EPS)));

            vec3 s = to_lin(texture(tex, vec2(sample_x, coord.y)).rgb);

            vec3 di = left_last - s;

            vec3 eff_s;
            for (int i = 0; i < 3; ++i) {
                if (left_updating[i] < 0.5) {
                    eff_s[i] = left_last[i];
                    continue;
                }
                if (abs(di)[i] > i_sigma) {
                    left_updating[i] = 0.0;
                    eff_s[i] = min(s[i], left_last[i]);
                    left_last[i] = eff_s[i];
                    continue;
                }
                eff_s[i] = s[i];
                left_last[i] = s[i];
            }
            vec3 w_i = vec3(1.0);

            vec3 w = w_x * w_i;
            sum += w;
            result += w * eff_s;
        }
    }
    // return min(center_i, result / sum);
    return result / sum;
}

void main() {
    // FragColor.rgb = to_lin(texture(Original, vTexCoord).rgb);
    // FragColor[3] = 1.0;
    // return

    // const float m = param.OutputSize.w * param.AA_STRENGTH * 100;

    // const vec2 axes = vec2(0.4, 0.3);
    // const vec2 center = vec2(0.5 + 0.0 * 0.785 * axes[0], 0.5);

    // const float fx = f(vTexCoord, center, axes);
    // const vec2 gradfx = gradf(vTexCoord, center, axes);
    // // const vec2 gradfx = vec2(dFdx(fx), dFdy(fx)) / m;
    // float dist = fx / length(gradfx);

    // // // Sanity check gradient
    // // if (vTexCoord.x < 0.3) {
    // //     dist = abs(vTexCoord.y - 0.5) - 0.2;
    // // }

    // // Draw shapes
    // FragColor = vec4(1.0 - smoothstep(-0.5 * m, 0.5 * m, dist), 0.0,
    // 0.0, 1.0);

    // // MSAA ellipse
    // FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    // const vec2 start =
    //     (floor(vTexCoord * param.OutputSize.xy) + 0.5) * param.OutputSize.zw;

    // const vec2 d3d_msaa_pattern[] = {{1, 1},   {-1, -3}, {-3, 2},  {4, -1},
    //                                  {-5, -2}, {2, 5},   {5, 3},   {3, -5},
    //                                  {-2, 6},  {0, -7},  {-4, -6}, {-6, 4},
    //                                  {-8, 0},  {7, -4},  {6, 7},   {-7, -8}};

    // float coverage = 0.0;
    // float max_coverage = 0.0;
    // for (int i = 0; i < 16; ++i) {
    //     const vec2 offset = d3d_msaa_pattern[i] / 16.0;
    //     const vec2 coord = start + offset * param.OutputSize.zw *
    //     param.AA_STRENGTH * 10; const float w = 1.0 - dot(offset, offset);
    //     max_coverage += w;

    //     const vec2 delta = (coord - center) / (axes + EPS);
    //     coverage += w * (1.0 - step(1.0, dot(delta, delta)));
    // }
    // FragColor.r = coverage / max_coverage;

    // Cross through middle
    // if (abs(vTexCoord.x - 0.5) < param.OutputSize.z ||
    //     abs(vTexCoord.y - 0.5) < param.OutputSize.w) {
    //     FragColor.rgb = vec3(1.0) - FragColor.rgb;
    // }

    // return;

    // vec3 col = (dist > 0.0) ? vec3(0.9, 0.6, 0.3) : vec3(0.65, 0.85, 1.0);
    // col *= 1.05 - exp(-6.0 * abs(dist));
    // col *= 0.8 + 0.2 * cos(110.0 * dist);
    // col = mix(col, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(dist)));
    // FragColor.rgb = col;
    // FragColor[3] = 1.0;

    // FragColor.rgb =
    //     sample_scanline_new(vec3(param.H1, 0.6, 0.3), vec3(param.H2, 0.0,
    //     0.5),
    //                         vec2(0.5), vTexCoord);
    // return;

    // Important coordinates
    const float px_upper =
        (vTexCoord.y - param.OutputSize.w * 0.5) * param.OriginalSize.y;
    const float px_lower = px_upper + param.OutputSize.w * param.OriginalSize.y;
    const vec2 tx_coord = vTexCoord * param.OriginalSize.xy;

    // Gaussian
    // const float coeffs[] = get_gaussian();

    // Sample nearest scanline
    // vec3 samples[] = {
    //     to_lin(
    //         texture(Source, param.OriginalSize.zw * (tx_coord - vec2(2.0,
    //         0.0)))
    //             .rgb) *
    //         param.MAX_THICK,
    //     to_lin(
    //         texture(Source, param.OriginalSize.zw * (tx_coord - vec2(1.0,
    //         0.0)))
    //             .rgb) *
    //         param.MAX_THICK,
    //     to_lin(texture(Source, param.OriginalSize.zw * (tx_coord)).rgb) *
    //         param.MAX_THICK,
    //     to_lin(
    //         texture(Source, param.OriginalSize.zw * (tx_coord + vec2(1.0,
    //         0.0)))
    //             .rgb) *
    //         param.MAX_THICK,
    //     to_lin(
    //         texture(Source, param.OriginalSize.zw * (tx_coord + vec2(2.0,
    //         0.0)))
    //             .rgb) *
    //         param.MAX_THICK};

    // const vec3 signal =
    //     to_lin(texture(Source, param.OriginalSize.zw * (tx_coord)).rgb) *
    //     param.MAX_THICK;

    const float scanline_center = floor(tx_coord.y) + 0.5;
    FragColor.rgb = sample_scanline(
        get_bilateral(Source, vTexCoord, param.H1, param.H2) * param.MAX_THICK,
        scanline_center, px_upper, px_lower);

    // const vec3 signal_1 =
    //     get_blended_value(samples, fract(tx_coord.x), 0.5) * param.MAX_THICK;
    // const float second_nearest_offset = fract(tx_coord.x) < 0.5 ? 1.0 : -1.0;
    // const vec3 signal_2 =
    //     get_blended_value(samples,
    //                       second_nearest_offset + 1.0 - fract(tx_coord.x),
    //                       second_nearest_offset + 0.5) *
    //     param.MAX_THICK;
    // const vec2 signal_center =
    //     vec2(fract(tx_coord.x) < 0.5 ? floor(tx_coord.x) : ceil(tx_coord.x),
    //          floor(tx_coord.y) + 0.5);

    // FragColor.rgb = sample_scanline_new(
    //     fract(tx_coord.x) < 0.5 ? signal_2 : signal_1,
    //     fract(tx_coord.x) < 0.5 ? signal_1 : signal_2, signal_center,
    //     tx_coord);

    // Sample 2nd-nearest scanline. We can just add the values.
    const float offset = mix(1.0, -1.0, step(tx_coord.y, scanline_center));
    FragColor.rgb += sample_scanline(
        get_bilateral(Source,
                      vTexCoord + vec2(0.0, offset * param.OriginalSize.w),
                      param.H1, param.H2) *
            param.MAX_THICK,
        scanline_center + offset, px_upper, px_lower);

    // Normalize value from per-texel to per-pixel
    FragColor.rgb *= param.OutputSize.y * param.OriginalSize.w;

    FragColor.rgb = to_srgb(FragColor.rgb);
    FragColor[3] = 1.0;
}
