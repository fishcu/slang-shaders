#version 450

/*
    Brightness-Optimized SDR Scanlines v1.0 by fishku
    Copyright (C) 2023
    Public domain license (CC0)

    Changelog:
    v1.0: Initial release.
*/

#include "parameters.slang"

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
    float INPUT_GAMMA;
    float H_BLUR;
    float V_BLUR;
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

vec3 to_lin(vec3 x) { return pow(x, vec3(param.INPUT_GAMMA)); }

vec3 to_srgb(vec3 x) { return pow(x, vec3(1.0 / 2.2)); }

float cubic_pulse(float x, float w) {
    x = abs(x);
    if (x >= w) {
        return 0.0;
    }
    x /= w;
    return 1.0 - x * x * (3.0 - 2.0 * x);
}

vec3 cubic_pulse(vec3 x, float w) {
    return vec3(cubic_pulse(x.x, w), cubic_pulse(x.y, w), cubic_pulse(x.z, w));
}

void main() {
    vec3 w_sum = vec3(1.0);
    const vec3 ci = to_lin(texture(Source, vTexCoord).rgb);
    FragColor.rgb = ci;
    float dx = 1.0;
    for (float offset_x = param.SourceSize.z;
         offset_x <= param.H_BLUR * param.SourceSize.z; offset_x += param.SourceSize.z) {
        vec2 offset = vec2(offset_x, 0.0);

        const float dx_w = cubic_pulse(dx, param.H_BLUR);

        vec3 i = to_lin(texture(Source, vTexCoord + offset).rgb);
        vec3 di_w = cubic_pulse(ci - i, param.V_BLUR);
        vec3 w = dx_w * di_w;
        w_sum += w;
        FragColor.rgb += i * w;

        i = to_lin(texture(Source, vTexCoord - offset).rgb);
        di_w = cubic_pulse(ci - i, param.V_BLUR);
        w = dx_w * di_w;
        w_sum += w;
        FragColor.rgb += i * w;

        dx += 1.0;
    }
    FragColor.rgb /= w_sum;

    FragColor.rgb = to_srgb(FragColor.rgb);

    FragColor.a = 1.0;
}
