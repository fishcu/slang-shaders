#version 450

#pragma parameter SUPERSAMPLING "Supersampling level" 4.0 0.0 8.0 0.1
#pragma parameter ROT "Rotation" 0.0 0.0 3.141 0.1
#pragma parameter ANIMATE "Animate" 0.0 0.0 1.0 1.0
#pragma parameter RNG1 "RNG Seed 1" 0.0 0.0 10000.0 1.0
#pragma parameter RNG2 "RNG Seed 2" 0.0 0.0 10000.0 1.0
#pragma parameter RNG3 "RNG Seed 3" 0.0 0.0 10000.0 1.0

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float SUPERSAMPLING;
    float ROT;
    float ANIMATE;
    float RNG1, RNG2, RNG3;
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

// https://www.shadertoy.com/view/4djSRW
vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

vec2 hash23(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

void main() {
    FragColor = vec4(vec3(0.0), 1.0);

    // Center of screen cross
    if ((abs(vTexCoord.x - 0.5) < 0.1 && abs(vTexCoord.y - 0.5) < 0.001) ||
        (abs(vTexCoord.x - 0.5) < 0.001 && abs(vTexCoord.y - 0.5) < 0.1)) {
        FragColor.rgb = vec3(1.0);
    }

    // Local texture space derivatives
    // mat2 dist = mat2(dFdx(uv), dFdy(uv));
    mat2 dist = mat2(vec2(0.9, 0.1), vec2(0.1, 0.9));

    vec3 rand = hash33(vec3(param.RNG1, param.RNG2, param.RNG3));
    // vec3 rand = hash33(vec3(vTexCoord * param.OutputSize.xy,
    //                         param.ANIMATE * float(param.FrameCount % 8191)));

    rand.z *= 2.0 * PI;
    float s = sin(param.ROT + rand.z);
    float c = cos(param.ROT + rand.z);
    mat2 rot = mat2(c, -s, s, c);
    float scale = max(abs(c - s), abs(s + c));

    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    const vec2 alpha = vec2(0.7548776662, 0.56984029);

    int num_samples = int(exp2(param.SUPERSAMPLING));
    int sample_count = 0;
    for (int i = 0; i < num_samples; ++i) {
        // Rejection sampling
        // vec2 offset = scale * rot *
        //               (fract(float(sample_count++) * alpha + rand.xy) - 0.5);
        // while (offset.x < -0.5 || offset.x > 0.5 || offset.y < -0.5 ||
        //        offset.y > 0.5) {
        //     offset = scale * rot *
        //              (fract(float(sample_count++) * alpha + rand.xy) - 0.5);
        // }

        // "polar coordinate transform"
        // vec2 offset = fract(float(sample_count++) * alpha + rand.xy);
        // vec2 polar = vec2(sqrt(offset.x), 2 * PI * offset.y);
        // offset = polar.x * vec2(cos(polar.y), sin(polar.y));
        // // Apply rotation
        // offset = rot * offset;

        // Box-Muller transform
        // vec2 offset = fract(float(sample_count++) * alpha + rand.xy);
        
        // vec2 offset = hash23(vec3(param.RNG1 + sample_count++, param.RNG2,
        // param.RNG3));
        
        vec2 offset = hash23(vec3(
            vTexCoord * param.OutputSize.xy,
            sample_count++ + param.ANIMATE * float(param.FrameCount %8191)));
        
        offset = 0.2 * sqrt(-2.0 * log(offset.x)) *
                 vec2(cos(2 * PI * offset.y), sin(2 * PI * offset.y));
        // Apply rotation
        offset = rot * offset;

        // // Rejection sampling for circle
        // vec2 offset = (fract(float(sample_count++) * alpha + rand.xy) - 0.5);
        // while (dot(offset, offset) > 0.25) {
        //     offset = (fract(float(sample_count++) * alpha + rand.xy) - 0.5);
        // }
        // // Apply rotation
        // offset = rot * offset;

        // Apply distortion
        offset = dist * offset;

        // Center on screen
        offset *= 0.5;
        offset += 0.5;

        FragColor.rgb = max(FragColor.rgb, circle(vTexCoord, offset, 0.01));
    }
}