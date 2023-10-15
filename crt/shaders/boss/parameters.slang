// clang-format off
#pragma parameter BOSS_SETTINGS "=== BOSS v1.0 settings ===" 0.0 0.0 1.0 1.0
// #include "crt-Ntap-common-params.slang"
#pragma parameter MIN_THICK "Scanline thickness of dark pixels" 0.0 0.0 1.4 0.05
#pragma parameter MAX_THICK "Scanline thickness of bright pixels" 1.0 0.0 1.4 0.05

#pragma parameter INPUT_GAMMA "Input gamma" 2.2 1.0 3.0 0.1
#pragma parameter OUTPUT_GAMMA "Output gamma" 2.2 1.0 3.0 0.1

#pragma parameter AA_STRENGTH "Scanline falloff width in pixels" 1.0 0.5 4.0 0.1

#pragma parameter H_BLUR "Horizontal smoothing strength" 0.0 0.0 10.0 0.1
#pragma parameter V_BLUR "Pixel value smoothing strength" 2.0 0.0 2.0 0.1
// clang-format on
