#version 450

#pragma parameter SUPERSAMPLING "Supersampling level" 4.0 0.0 8.0 0.1

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float SUPERSAMPLING;
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
    FragColor = vec4(vec3(0.0), 1.0);
    int num_samples = int(exp2(filtering_level));
    
        // Generate a random angle for this fragment
    // hash function from: https://www.shadertoy.com/view/4djSRW
    rng_seed = fract(rng_seed * .1031);
    rng_seed += dot(rng_seed, rng_seed.zyx + 31.32);
    float random_angle =
        fract((rng_seed.x + rng_seed.y) * rng_seed.z) * 6.28318530718;  // 2*pi
    mat2 rot = mat2(cos(random_angle), -sin(random_angle), sin(random_angle),
                    cos(random_angle));

    mat2 tf = ellipse_approx * rot;

    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    const vec2 alpha = vec2(0.7548776662, 0.56984029);

    for (int i = 0; i < num_samples; i++) {
        // Generate rotated R2 sequence sample
        vec2 offset = fract(float(i) * alpha) - 0.5;

        // Calculate weight
        // Simplified cubic pulse (assuming center=0, w=sqrt(2)/2)
        float t = length(offset) * 1.41421;
        float weight = 1.0 - t * t * (3.0 - 2.0 * t);

        // Rotate and distort to ellipse
        offset = tf * offset;

        // Implement blending between mip-map levels in software because current
        // implementation is broken. See:
        // https://github.com/libretro/RetroArch/issues/16567;
        color +=
            mix(textureLod(tex, uv + offset, mip_level_i),
                textureLod(tex, uv + offset, mip_level_i + 1.0), mip_level_f) *
            weight;
        total_weight += weight;
    }

    return color / total_weight;
}