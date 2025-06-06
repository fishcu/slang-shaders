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

void main() {
    vec2 period;
    const vec2 phase = modf(tx_coord - 0.5, period);

    const vec2 tx_to_uv = param.SourceSize.zw;
    vec3 samples[] = {texture(Source, (period + vec2(0.5, 0.5)) * tx_to_uv).rgb,
                      texture(Source, (period + vec2(1.5, 0.5)) * tx_to_uv).rgb,
                      texture(Source, (period + vec2(0.5, 1.5)) * tx_to_uv).rgb,
                      texture(Source, (period + vec2(1.5, 1.5)) * tx_to_uv).rgb};

    const vec3 col = texture(Source, vTexCoord).rgb;
    vec3 col_contrast_h = abs(dFdx(col));
    vec3 col_contrast_v = abs(dFdy(col));

    // Fudge effect to reduce shimmering
    col_contrast_h = pow(col_contrast_h, vec3(param.PIX_AA_LUMA_SHARPENING));
    col_contrast_v = pow(col_contrast_v, vec3(param.PIX_AA_LUMA_SHARPENING));

    vec2 trans_width_base = param.SourceSize.xy * param.OutputSize.zw;
    vec2 t_r = clamp(
        (phase - 0.5) / (trans_width_base * vec2(col_contrast_h.r, col_contrast_v.r) + 1e-4) + 0.5,
        0.0, 1.0);
    vec2 t_g = clamp(
        (phase - 0.5) / (trans_width_base * vec2(col_contrast_h.g, col_contrast_v.g) + 1e-4) + 0.5,
        0.0, 1.0);
    vec2 t_b = clamp(
        (phase - 0.5) / (trans_width_base * vec2(col_contrast_h.b, col_contrast_v.b) + 1e-4) + 0.5,
        0.0, 1.0);

    vec3 res;
    res.r = mix(mix(samples[0].r, samples[1].r, t_r.x),
                mix(samples[2].r, samples[3].r, t_r.x), t_r.y);
    res.g = mix(mix(samples[0].g, samples[1].g, t_g.x),
                mix(samples[2].g, samples[3].g, t_g.x), t_g.y);
    res.b = mix(mix(samples[0].b, samples[1].b, t_b.x),
                mix(samples[2].b, samples[3].b, t_b.x), t_b.y);

    FragColor = vec4(pow(res, vec3(1.0 / 2.2)), 1.0);
}
