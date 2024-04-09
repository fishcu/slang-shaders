#version 450

/*
    Brightness-Optimized SDR Scanlines v1.0 by fishku
    Copyright (C) 2023
    Public domain license (CC0)

    Changelog:
    v1.0: Initial release.
*/

// clang-format off
#pragma parameter BOSS_SETTINGS "=== BOSS v1.0 settings ===" 0.0 0.0 1.0 1.0

#pragma parameter MIN_THICK "Scanline thickness of dark pixels" 0.0 0.0 1.4 0.05
#pragma parameter MAX_THICK "Scanline thickness of bright pixels" 1.0 0.0 1.4 0.05

#pragma parameter SUBPX_POS "Scanline subpixel position" 0.5 0.0 1.0 0.01

// TODO replace with input and output gamma settings
#pragma parameter S_GAMMA_OFFS "Signal gamma offset" 0.0 -2.0 2.0 0.1

#pragma parameter AA_STRENGTH "Smoothing strength" 1.0 0.5 4.0 0.1
// #pragma parameter H_BLUR "Horizontal blur strength" 0.3 0.0 3.0 0.05

// #pragma parameter H1 "Signal 1" 0.0 0.0 5.0 0.05
// #pragma parameter H2 "Signal 2" 0.0 0.0 5.0 0.05
// #pragma parameter ELLIPSE_W "Ellipse width" 1.0 0.0 3.0 0.05
// clang-format on

#define NUMERICAL_FUDGE 1.e-3

layout(push_constant) uniform Push {
    vec4 OriginalSize;
    vec4 SourceSize;
    vec4 OutputSize;
    float MIN_THICK;
    float MAX_THICK;
    float SUBPX_POS;
    float S_GAMMA_OFFS;
    float AA_STRENGTH;
    // float H_BLUR;
    //
    // float H1;
    // float H2;
    // float ELLIPSE_W;
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

vec3 to_lin(vec3 x) { return pow(x, vec3(2.2 + param.S_GAMMA_OFFS)); }

vec3 to_srgb(vec3 x) { return pow(x, vec3(1.0 / 2.2)); }

vec2 perp(vec2 a) { return vec2(a.y, -a.x); }

float cro(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }

float line_seg_sdf(in vec2 p, in vec2 q, in vec2 x) {
    vec2 xp = x - p;
    vec2 e = q - p;
    float h = clamp(dot(xp, e) / dot(e, e), 0.0, 1.0);
    float d = length(xp - e * h);
    float s = 1.0;
    // even-odd rule
    if ((xp.x > 0.0) != (xp.x > e.x)) {
        if ((e.x * xp.y < e.y * xp.x) != (e.x < 0.0)) {
            s = -s;
        }
    }
    return d * s;
}

// Get SDF of circle arc while first constructing arc from two points and
// tangent vector
float circle_arc_sdf(vec2 p, vec2 q, vec2 t, vec2 x) {
    vec2 n = perp(t);
    vec2 d = q - p;
    float lambda = 0.5 * dot(d, d) / dot(n, d);
    vec2 c = p + lambda * n;
    float r2 = lambda * lambda * dot(n, n);
    // Early out: If circle is very large, return line SDF.
    if (r2 > 1.e8) {
        return line_seg_sdf(p, q, x);
    }
    // If point is inside cone (p, c, q), return min dist. to p & q
    // else, return distance to radius.
    p -= c;
    q -= c;
    x -= c;
    // Figure out sign of SDF by using even-odd rule.
    // Redefine n to be the bisector of the triangle (p, c, q).
    n = lambda * perp(d);
    // This is missing |n|*|p| = |n|*r, but it often cancels out.
    float cos_opening_angle = dot(n, p);
    float s = 1.0;
    float y_on_circle = r2 - x.x * x.x;
    if (y_on_circle >= 0.0) {
        // This implies abs(x.x) < r.
        y_on_circle = sqrt(y_on_circle);
        // Check if line drawn straight from x to infinity
        // crosses the arc zero, one, or two times by checking if
        // intersection points of circle with line are on arc.
        // alpha < beta => cos(alpha) > cos(beta)
        if (x.y < -y_on_circle &&
            dot(n, vec2(x.x, -y_on_circle)) < cos_opening_angle) {
            s = -s;
        }
        if (x.y < y_on_circle &&
            dot(n, vec2(x.x, y_on_circle)) < cos_opening_angle) {
            s = -s;
        }
    }
    float dist_xc = length(x);
    float r = sqrt(r2);
    // Here's the only instance where the vector lengths in the
    // comparison of dot products doesn't cancel out.
    if (dot(n, x) * r < cos_opening_angle * dist_xc) {
        return abs(dist_xc - r) * s;
    }
    vec2 xa = x - p;
    vec2 xb = x - q;
    return sqrt(min(dot(xa, xa), dot(xb, xb))) * s;
}

float biarc_sdf(vec2 p, vec2 q, vec2 t0, vec2 t1, vec2 x) {
    t0 /= length(t0);
    t1 /= length(t1);

    // if (x.x > 0.5) {
    //     FragColor.a = 1.0;
    //     FragColor.b = 1.0;
    // }

    // chord given by points on circle
    vec2 d = p - q;
    // vector along which center must lie
    vec2 n = perp(d);
    float denom = dot(n, t0 - t1);
    vec2 j;
    if (abs(denom) < 1.0e-6) {
        // Both tangent vectors are the same.
        j = 0.5 * (p + q);
        // FragColor.a = 1.0;
        // FragColor.r = 1.0;
    } else {
        // center of circle describing locus of joint points
        vec2 c = 0.5 * ((p + q) + dot(d, t0 + t1) / denom * n);

        // radius squared of circle describing locus of joint points
        vec2 pc = p - c;
        float r2 = dot(pc, pc);

        // Joint point is chosen as intersection of chord bisector with circle.
        // The closer one is chosen, which gives good results for the tangents
        // we care about.
        j = c + sign(cro(pc, d)) * sqrt(r2) * n / length(n);
    }

    // if (distance(x, vec2(0.0)) < 0.03) {
    //     FragColor.a = 1.0;
    //     FragColor.b = 1.0;
    // }

    // if (distance(x, j) < 0.02) {
    //     FragColor.a = 1.0;
    //     FragColor.b = 1.0;
    // }

    // if (distance(x, p) < 0.03) {
    //     FragColor.a = 1.0;
    //     FragColor.r = 1.0;
    // }
    // if (distance(x, q) < 0.03) {
    //     FragColor.a = 1.0;
    //     FragColor.g = 1.0;
    // }

    // if (abs(line_seg_sdf(p, p + t0 * 0.9, x)) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.r = 1.0;
    // }
    // if (abs(line_seg_sdf(q, q - t1 * 0.9, x)) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.g = 1.0;
    // }

    // Find arcs and evaluate SDF in one go
    float sd1 = circle_arc_sdf(p, j + vec2(NUMERICAL_FUDGE, 0.0), t0, x);
    float sd2 = circle_arc_sdf(q, j - vec2(NUMERICAL_FUDGE, 0.0), -t1, x);
    float sd = min(abs(sd1), abs(sd2));
    float s = sign(sd1 * sd2);

    // There should not be any frags inside both shapes.
    // This accounts for numerical fudging of borders.
    if (sd1 < 0.0 && sd2 < 0.0) {
        s = -1.0;
    }

    return sd * s;
}

float get_slope(float l, float c, float r) {
    // if ((c - l) * (r - c) < -1.e-2) {
    //     // inflection
    //     return 0.0;
    // }
    return 0.5 * (r - l);
}

vec3 get_slope(vec3 l, vec3 c, vec3 r) {
    // TODO: optimize later
    vec3 res;
    for (int i = 0; i < 3; ++i) {
        res[i] = get_slope(l[i], c[i], r[i]);
    }
    return res;
}

void main() {
    const vec2 tx_coord = vTexCoord * param.OriginalSize.xy;
    vec2 itx_coord;
    vec2 ftx_coord = modf(tx_coord, itx_coord);

    // const float scanline_center = floor(tx_coord.y) + param.SUBPX_POS;

    const vec3 smp[] = {
        0.4 * to_lin(texture(Source, (itx_coord + vec2(-1.5, 0.5)) *
                                         param.OriginalSize.zw)
                         .rgb),
        0.4 * to_lin(texture(Source, (itx_coord + vec2(-0.5, 0.5)) *
                                         param.OriginalSize.zw)
                         .rgb),
        0.4 * to_lin(texture(Source, (itx_coord + vec2(0.5, 0.5)) *
                                         param.OriginalSize.zw)
                         .rgb),
        0.4 * to_lin(texture(Source, (itx_coord + vec2(1.5, 0.5)) *
                                         param.OriginalSize.zw)
                         .rgb),
        0.4 * to_lin(texture(Source, (itx_coord + vec2(2.5, 0.5)) *
                                         param.OriginalSize.zw)
                         .rgb)};
    const vec3 slopes[] = {get_slope(smp[0], smp[1], smp[2]),
                           get_slope(smp[1], smp[2], smp[3]),
                           get_slope(smp[2], smp[3], smp[4])};

    // Debug
    FragColor = vec4(0.0);

    // if (ftx_coord.x < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.rgb = vec3(1.0);
    // }

    // values
    // ftx_coord.y = 0.5 - ftx_coord.y;
    // if (ftx_coord.x < 0.16 && abs(ftx_coord.y - smp[0][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.g = 1.0;
    // }
    // if (ftx_coord.x < 0.33 && abs(ftx_coord.y - smp[1][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.r = 1.0;
    // }
    // if (ftx_coord.x >= 0.33 && ftx_coord.x < 0.66 &&
    //     abs(ftx_coord.y - smp[2][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.g = 1.0;
    // }
    // if (ftx_coord.x >= 0.66 && abs(ftx_coord.y - smp[3][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.b = 1.0;
    // }
    // if (ftx_coord.x >= 0.86 && abs(ftx_coord.y - smp[4][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.r = 1.0;
    // }
    // ftx_coord.y = 0.5 - ftx_coord.y;

    // slopes
    // ftx_coord.y = 1.0 - ftx_coord.y;
    // if (ftx_coord.x < 0.33 && abs(ftx_coord.y - 0.5 - slopes[0][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.r = 1.0;
    // }
    // if (ftx_coord.x >= 0.33 && ftx_coord.x < 0.66 &&
    //     abs(ftx_coord.y - 0.5 - slopes[1][0]) < 0.01) {
    //     FragColor.a = 1.0;
    //     FragColor.g = 1.0;
    // }
    // if (ftx_coord.x >= 0.66 && abs(ftx_coord.y - 0.5 - slopes[2][0]) < 0.01)
    // {
    //     FragColor.a = 1.0;
    //     FragColor.b = 1.0;
    // }
    // ftx_coord.y = 1.0 - ftx_coord.y;

    // Evaluate two biarcs that are closest to current pixel
    // TODO vectorize
    ftx_coord.y = abs(ftx_coord.y - 0.5);
    // ftx_coord.y -= 0.3;
    vec2 tx_per_px = param.OutputSize.zw * param.OriginalSize.xy;
    for (int i = 0; i < 3; ++i) {
        float sd1 = biarc_sdf(vec2(-0.5 - NUMERICAL_FUDGE, smp[1][i]),  //
                              vec2(0.5 + NUMERICAL_FUDGE, smp[2][i]),   //
                              vec2(1.0, slopes[0][i]),                  //
                              vec2(1.0, slopes[1][i]),                  //
                              ftx_coord);

        float sd2 = biarc_sdf(vec2(0.5 - NUMERICAL_FUDGE, smp[2][i]),  //
                              vec2(1.5 + NUMERICAL_FUDGE, smp[3][i]),  //
                              vec2(1.0, slopes[1][i]),                 //
                              vec2(1.0, slopes[2][i]),                 //
                              ftx_coord);

        // float sd1 = line_seg_sdf(vec2(-0.5, smp[1][i]), vec2(0.5, smp[2][i]),
        //                          ftx_coord);
        // float sd2 =
        //     line_seg_sdf(vec2(0.5, smp[2][i]), vec2(1.5, smp[3][i]),
        //     ftx_coord);

        float d = min(abs(sd1), abs(sd2));
        float s = sign(sd1 * sd2);
        // There should not be any frags inside both shapes.
        // This accounts for numerical fudging of borders.
        if (sd1 < 0.0 && sd2 < 0.0) {
            s = -1.0;
        }

        const float aa_thickness = tx_per_px.x * param.AA_STRENGTH;
        if (FragColor.a == 0.0) {
            FragColor[i] = 1.0 - smoothstep(-0.5 * aa_thickness,
                                            0.5 * aa_thickness, s * d);
            // FragColor[i] = 1.0 - step(0.0, sd);
        }
    }

    FragColor.rgb = to_srgb(FragColor.rgb);
    FragColor.a = 1.0;
}
