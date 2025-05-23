#version 450

/*
   CRT - Guest - Advanced - HD - Pass1
   
   Copyright (C) 2018-2025 guest(r)

   Incorporates many good ideas and suggestions from Dr. Venom.
   I would also like give thanks to many Libretro forums members for continuous feedback, suggestions and using the shader.
   
   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
   
*/

layout(push_constant) uniform Push
{
	vec4 SourceSize;
	vec4 OriginalSize;
	vec4 OutputSize;
	uint FrameCount;
	float SIGMA_HOR;
	float HSHARPNESS;
	float S_SHARP;
	float HARNG;
	float HSHARP; 
	float spike;
	float SIGMA_VER;
	float VSHARPNESS;
	float internal_res, auto_res;
	float MAXS;
} params;

layout(std140, set = 0, binding = 0) uniform UBO
{
	mat4 MVP;
} global;


#pragma parameter bogus_filtering "[ FILTERING OPTIONS ]: " 0.0 0.0 1.0 1.0

#pragma parameter auto_res "          SNES/Amiga Hi-Res Auto Mode" 0.0 0.0 1.0 1.0
#define auto_res params.auto_res

#pragma parameter internal_res "          Internal Resolution" 1.0 0.5 8.0 0.10
float internal_res = mix(params.internal_res, 1.75*params.internal_res, clamp(auto_res * round(params.OriginalSize.x/300.0)-1.0, 0.0, 1.0));

#pragma parameter HSHARPNESS "          Horizontal Filter Range" 1.0 1.0 8.0 0.05
#define HSHARPNESS params.HSHARPNESS

#pragma parameter SIGMA_HOR "          Horizontal Blur Sigma" 0.50 0.1 7.0 0.025
#define SIGMA_HOR params.SIGMA_HOR

#pragma parameter S_SHARP "          Substractive Sharpness" 1.0 0.0 2.0 0.10
#define S_SHARP params.S_SHARP

#pragma parameter HSHARP "          Sharpness Definition" 1.20 0.0 2.0 0.10
#define HSHARP params.HSHARP

#pragma parameter MAXS "          Maximum Sharpness" 0.15 0.0 0.30 0.01
#define MAXS params.MAXS

#pragma parameter HARNG "          Substractive Sharpness Ringing" 0.2 0.0 4.0 0.10
#define HARNG params.HARNG 

#pragma parameter bogus_vfiltering "[ VERTICAL/INTERLACING FILTERING OPTIONS ]: " 0.0 0.0 1.0 1.0

#pragma parameter VSHARPNESS "          Vertical Filter Range" 1.0 1.0 8.0 0.05
#define VSHARPNESS global.VSHARPNESS

#pragma parameter SIGMA_VER "          Vertical Blur Sigma" 0.50 0.1 7.0 0.025
#define SIGMA_VER global.SIGMA_VER

#pragma parameter spike "          Scanline Spike Removal" 1.0 0.0 2.0 0.10
#define spike params.spike


#define COMPAT_TEXTURE(c,d) texture(c,d)
#define TEX0 vTexCoord

#define OutputSize params.OutputSize
#define gl_FragCoord (vTexCoord * OutputSize.xy)

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;

void main()
{
   gl_Position = global.MVP * Position;
   vTexCoord = TexCoord * 1.00001;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D LinearizePass;

float invsqrsigma = 1.0/(2.0*SIGMA_HOR*SIGMA_HOR*internal_res*internal_res);

float gaussian(float x)
{
	return exp(-x*x*invsqrsigma);
}

void main()
{	
	vec2 prescalex = vec2(textureSize(LinearizePass, 0)) / params.OriginalSize.xy;

	vec4 SourceSize = params.OriginalSize * vec4(prescalex.x, prescalex.y, 1.0/prescalex.x, 1.0/prescalex.y);
	
	float f = fract(SourceSize.x * vTexCoord.x);
	f = 0.5 - f;
	vec2 tex = floor(SourceSize.xy * vTexCoord)*SourceSize.zw + 0.5*SourceSize.zw;
	vec3 color = 0.0.xxx;
	float scolor = 0.0;
	vec2 dx  = vec2(SourceSize.z, 0.0);

	float w = 0.0;
	float swsum = 0.0;
	float wsum = 0.0;
	vec3 pixel;

	float hsharpness = HSHARPNESS * internal_res;
	vec3 cmax = 0.0.xxx;
	vec3 cmin = 1.0.xxx;
	float sharp = gaussian(hsharpness) * S_SHARP;
	float maxsharp = MAXS;
	float FPR = hsharpness;
	float FPRi = 1.0/hsharpness;
	float fpx = 0.0;
	float sp = 0.0;
	float sw = 0.0;

	float ts = 0.025;
	vec3 luma = vec3(0.2126, 0.7152, 0.0722); 
	float LOOPSIZE = ceil(2.0*FPR);
	
	float n = -LOOPSIZE;
	
	do
	{
		pixel  = COMPAT_TEXTURE(LinearizePass, tex + n*dx).rgb;
		
		w = gaussian(n+f) - sharp;
		fpx = (abs(n+f)-FPR)*FPRi;
		if (w < 0.0) w = max(w, mix(-maxsharp, 0.0, pow(clamp(fpx,0.0,1.0), HSHARP)));
		else
		{
			cmax = max(cmax, pixel); cmin = min(cmin, pixel);
			sw = w * (dot(pixel,luma) + ts); 
			sp = max(max(pixel.r,pixel.g),pixel.b);
			scolor = scolor + sw * sp;
			swsum = swsum + sw;			
		}
		color = color + w * pixel;
		wsum  = wsum + w;

		n = n + 1.0;
			
	} while (n <= LOOPSIZE);

	color = color / wsum;
	scolor = scolor / swsum;
	
	color = clamp(mix(clamp(color, cmin, cmax), color, HARNG), 0.0, 1.0);
	
	scolor = clamp(mix(max(max(color.r, color.g),color.b), scolor, spike), 0.0, 1.0);
	
	FragColor = vec4(color, scolor);
} 