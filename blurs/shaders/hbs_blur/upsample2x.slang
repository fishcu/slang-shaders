#version 450

// See main shader file for copyright and other information.

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
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
    /*
    Every output fragment is the weighted average of 4 input texels:
    [[1 3]
     [3 9]]
    The output fragment coordinate is inside the "9", and the other weights
    correspond to the three nearest texels.
    Assuming that the output size is exactly twice the input size, it turns out
    that the correct sampling location for this weight distribution is exactly
    the output fragment center, meaning we can skip any complex logic!
    */
    FragColor = vec4(texture(Source, vTexCoord).rgb, 1.0);
}
