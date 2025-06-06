#version 450

// See main shader file for copyright and other information.

#pragma parameter PIX_AA_LUMA_SHARPENING "Contrast-based sharpening" 0.2 0.0 1.0 0.1

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
    float PIX_AA_LUMA_SHARPENING;
}
param;

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; }
global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 tx_coord;
layout(location = 1) out vec2 vTexCoord;

void main() {
    gl_Position = global.MVP * Position;
    tx_coord = TexCoord * param.SourceSize.xy;
    vTexCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec2 tx_coord;
layout(location = 1) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;

float lin_srgb_to_lab_luma_approx(vec3 rgb) {
    // Convert linear sRGB to CIELAB L* (luma) using optimized approximation
    // Output is normalized to [0, 1] range
    float xyz_y = dot(rgb, vec3(0.2126729, 0.7151522, 0.0721750));
    return -0.27826705 * xyz_y + 1.27475151 * sqrt(xyz_y);
}

void main() {
    vec2 period;
    const vec2 phase = modf(tx_coord - 0.5, period);

    const vec2 tx_to_uv = param.SourceSize.zw;
    vec3 samples[] = {texture(Source, (period + vec2(0.5, 0.5)) * tx_to_uv).rgb,
                      texture(Source, (period + vec2(1.5, 0.5)) * tx_to_uv).rgb,
                      texture(Source, (period + vec2(0.5, 1.5)) * tx_to_uv).rgb,
                      texture(Source, (period + vec2(1.5, 1.5)) * tx_to_uv).rgb};

    // float lumas[] = {
    //     lin_srgb_to_lab_luma_approx(samples[0]), lin_srgb_to_lab_luma_approx(samples[1]),
    //     lin_srgb_to_lab_luma_approx(samples[2]), lin_srgb_to_lab_luma_approx(samples[3])};

    // vec2 luma_contrasts[] = {vec2(abs(lumas[0] - lumas[1]), abs(lumas[0] - lumas[2])),
    //                          vec2(abs(lumas[2] - lumas[3]), abs(lumas[1] - lumas[3]))};
    // vec2 luma_contrast = mix(luma_contrasts[0], luma_contrasts[1], phase);
    // vec2 luma_contrast = mix(luma_contrasts[0], luma_contrasts[1], 0.5);
    // vec2 luma_contrast = vec2(max(abs(lumas[0] - lumas[1]), abs(lumas[0] - lumas[2])),
    //                           max(abs(lumas[2] - lumas[3]), abs(lumas[1] - lumas[3])));

    // Different approach: central luma
    const float px_luma = lin_srgb_to_lab_luma_approx(texture(Source, vTexCoord).rgb);
    vec2 luma_contrast = abs(vec2(dFdx(px_luma), dFdy(px_luma)));

    // Fudge effect to reduce shimmering
    luma_contrast = pow(luma_contrast, vec2(param.PIX_AA_LUMA_SHARPENING));

    vec2 trans_width = param.SourceSize.xy * param.OutputSize.zw * luma_contrast;
    vec2 t = clamp((phase - 0.5) / (trans_width + 1e-4) + 0.5, 0.0, 1.0);
    // const vec2 offset = t * t * (3.0 - 2.0 * t);
    vec2 offset = t;

    vec3 res =
        mix(mix(samples[0], samples[1], offset.x), mix(samples[2], samples[3], offset.x), offset.y);

    FragColor = vec4(pow(res, vec3(1.0 / 2.2)), 1.0);
}
