#version 450

// See main shader file for copyright and other information.

layout(push_constant) uniform Push { vec4 SourceSize; }
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
    /*
    Every output fragment is the weighted average of 16 input texels:
    [[1 3 3 1]
     [3 9 9 3]
     [3 9 9 3]
     [1 3 3 1]]
    The output fragment coordinate is in the center of the 4x "9" block.
    This can be done efficiently with 4 bilinear taps without any complex
    coordinate maths, assuming that the output size is exactly half the input size.
    */
    const vec2 offset = param.SourceSize.zw * 0.75;
    FragColor =
        vec4((texture(Source, vTexCoord - offset).rgb +
              texture(Source, vTexCoord + vec2(offset.x, -offset.y)).rgb +
              texture(Source, vTexCoord + vec2(-offset.x, offset.y)).rgb +
              texture(Source, vTexCoord + offset).rgb) *
                 0.25,
             1.0);
}
