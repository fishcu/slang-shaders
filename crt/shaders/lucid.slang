#version 450

#pragma parameter SUPERSAMPLING "Supersampling level" 4.0 0.0 8.0 0.1
#pragma parameter ROT "Rotation" 0.0 0.0 3.141 0.1

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float SUPERSAMPLING;
    float ROT;
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

vec3 circle(vec2 x, vec2 c, float r) {
    vec2 dx = x - c;
    float d = sqrt(dot(dx, dx));
    return vec3(smoothstep(-0.001, 0.001, r - d), 0.0, 0.0);
}

#define PI 3.14159265

void main() {
    FragColor = vec4(vec3(0.0), 1.0);

    // Center of screen cross
    if ((abs(vTexCoord.x - 0.5) < 0.1 && abs(vTexCoord.y - 0.5) < 0.001) ||
        (abs(vTexCoord.x - 0.5) < 0.001 && abs(vTexCoord.y - 0.5) < 0.1)) {
        FragColor.rgb = vec3(1.0);
    }

    float s = sin(param.ROT);
    float c = cos(param.ROT);
    mat2 rot = mat2(c, -s, s, c);
    float enclosing_side = max(abs(c - s), abs(s + c));

    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    const vec2 alpha = vec2(0.7548776662, 0.56984029);

    int num_samples = int(exp2(param.SUPERSAMPLING));
    for (int i = 0, num_samples_done = 0; num_samples_done < num_samples; ++i) {
        vec2 offset = rot * (enclosing_side * (fract(float(i) * alpha + 0.5) - 0.5));

        // Rejection sampling
        while (offset.x < -0.5 || offset.x > 0.5 || offset.y < -0.5 ||
               offset.y > 0.5) {
            ++i;
            offset = rot * (enclosing_side * (fract(float(i) * alpha + 0.5) - 0.5));
        }

        // Center on screen
        offset *= 0.5;
        offset += 0.5;

        FragColor.rgb = max(FragColor.rgb, circle(vTexCoord, offset, 0.01));

        ++num_samples_done;
    }
}