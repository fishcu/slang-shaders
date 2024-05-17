#version 450

#include "parameters.inc"
// #include "shared.inc"

#include "../../../include/colorspace-tools.h"

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
layout(set = 0, binding = 2) uniform sampler2D Source;

void main() {
    const vec3 rgb = texture(Source, vTexCoord).rgb;
    FragColor = vec4(vec3(sqrt(fwidth(luma(rgb * rgb)))), 1.0);
}
