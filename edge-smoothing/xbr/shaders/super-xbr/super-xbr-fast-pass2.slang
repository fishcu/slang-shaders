#version 450

/*
   
  *******  Super XBR Shader - pass2  *******  This pass is optional.
   
  Copyright (c) 2022 Hyllian - sergiogdb@gmail.com

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

*/

layout(push_constant) uniform Push
{
	vec4 SourceSize;
	vec4 OriginalSize;
	vec4 OutputSize;
	uint FrameCount;
} params;

#define XBR_WEIGHT_P2     1.0
#define XBR_ANTI_RINGING  1.0
#define XBR_EDGE_STR_P2   2.0
#define weight           (XBR_WEIGHT_P2*1.29633/10.0)

const float wp1	= 1.0;
const float wp2	= 0.0;
const float wp3	= 0.0;
const float wp4	=-4.0;
const float wp5	= 0.0;
const float wp6	= 1.0;

layout(std140, set = 0, binding = 0) uniform UBO
{
	mat4 MVP;
} global;

const vec3 Y = vec3(.2126, .7152, .0722);

float RGBtoYUV(vec3 color)
{
  return dot(color, Y);
}

float df(float A, float B)
{
  return abs(A-B);
}

/*
                              P1
     |P0|B |C |P1|         C     F4          |a0|b1|c2|d3|
     |D |E |F |F4|      B     F     I4       |b0|c1|d2|e3|   |e1|i1|i2|e2|
     |G |H |I |I4|   P0    E  A  I     P3    |c0|d1|e2|f3|   |e3|i3|i4|e4|
     |P2|H5|I5|P3|      D     H     I5       |d0|e1|f2|g3|
                           G     H5
                              P2
*/

float d_wd(float wp1, float wp2, float wp3, float wp4, float wp5, float wp6, float b0, float b1, float c0, float c1, float c2, float d0, float d1, float d2, float d3, float e1, float e2, float e3, float f2, float f3)
{
	return (wp1*(df(c1,c2) + df(c1,c0) + df(e2,e1) + df(e2,e3)) + wp2*(df(d2,d3) + df(d0,d1)) + wp3*(df(d1,d3) + df(d0,d2)) + wp4*df(d1,d2) + wp5*(df(c0,c2) + df(e1,e3)) + wp6*(df(b0,b1) + df(f2,f3)));
}


vec3 min4(vec3 a, vec3 b, vec3 c, vec3 d)
{
    return min(a, min(b, min(c, d)));
}
vec3 max4(vec3 a, vec3 b, vec3 c, vec3 d)
{
    return max(a, max(b, max(c, d)));
}

float max4float(float a, float b, float c, float d)
{
    return max(a, max(b, max(c, d)));
}

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) out vec4 t1;
layout(location = 2) out vec4 t2;
layout(location = 3) out vec4 t3;
layout(location = 4) out vec4 t4;

void main()
{
   gl_Position = global.MVP * Position;
   vTexCoord = TexCoord * 1.0001;
   
   float dx = params.SourceSize.z;
   float dy = params.SourceSize.w;
   t1 = vTexCoord.xyxy + vec4(-2.0*dx, -2.0*dy, dx, dy);
   t2 = vTexCoord.xyxy + vec4(    -dx, -2.0*dy,  0, dy);
   t3 = vTexCoord.xyxy + vec4(-2.0*dx,     -dy, dx,  0);
   t4 = vTexCoord.xyxy + vec4(    -dx,     -dy,  0,  0);
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) in vec4 t1;
layout(location = 2) in vec4 t2;
layout(location = 3) in vec4 t3;
layout(location = 4) in vec4 t4;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;

void main()
{
	vec4 P0_ = texture(Source, t1.xy); vec3 P0 = P0_.rgb; float p0 = P0_.a;
	vec4 P1_ = texture(Source, t1.zy); vec3 P1 = P1_.rgb; float p1 = P1_.a;
	vec4 P2_ = texture(Source, t1.xw); vec3 P2 = P2_.rgb; float p2 = P2_.a;
	vec4 P3_ = texture(Source, t1.zw); vec3 P3 = P3_.rgb; float p3 = P3_.a;

	vec4  B_ = texture(Source, t2.xy); vec3  B =  B_.rgb; float  b =  B_.a;
	vec4  C_ = texture(Source, t2.zy); vec3  C =  C_.rgb; float  c =  C_.a;
	vec4 H5_ = texture(Source, t2.xw); vec3 H5 = H5_.rgb; float h5 = H5_.a;
	vec4 I5_ = texture(Source, t2.zw); vec3 I5 = I5_.rgb; float i5 = I5_.a;

	vec4  D_ = texture(Source, t3.xy); vec3  D =  D_.rgb; float  d =  D_.a;
	vec4 F4_ = texture(Source, t3.zy); vec3 F4 = F4_.rgb; float f4 = F4_.a;
	vec4  G_ = texture(Source, t3.xw); vec3  G =  G_.rgb; float  g =  G_.a;
	vec4 I4_ = texture(Source, t3.zw); vec3 I4 = I4_.rgb; float i4 = I4_.a;

	vec4  E_ = texture(Source, t4.xy); vec3  E =  E_.rgb; float  e =  E_.a;
	vec4  F_ = texture(Source, t4.zy); vec3  F =  F_.rgb; float  f =  F_.a;
	vec4  H_ = texture(Source, t4.xw); vec3  H =  H_.rgb; float  h =  H_.a;
	vec4  I_ = texture(Source, t4.zw); vec3  I =  I_.rgb; float  i =  I_.a;

/*
                              P1
     |P0|B |C |P1|         C     F4          |a0|b1|c2|d3|
     |D |E |F |F4|      B     F     I4       |b0|c1|d2|e3|   |e1|i1|i2|e2|
     |G |H |I |I4|   P0    E  A  I     P3    |c0|d1|e2|f3|   |e3|i3|i4|e4|
     |P2|H5|I5|P3|      D     H     I5       |d0|e1|f2|g3|
                           G     H5
                              P2
*/
	/* Calc edgeness in diagonal directions. */
	float d_edge  = (d_wd( wp1, wp2, wp3, wp4, wp5, wp6, d, b, g, e, c, p2, h, f, p1, h5, i, f4, i5, i4 ) - d_wd( wp1, wp2, wp3, wp4, wp5, wp6, c, f4, b, f, i4, p0, e, i, p3, d, h, i5, g, h5 ));

	float limits = XBR_EDGE_STR_P2 + 0.000001;
	
	/* Filter weights. Two taps only. */
	vec4 fw = vec4(-weight, weight+0.5, weight+0.5, -weight);

	/* Filtering and normalization in four direction generating four colors. */
	vec3 c1 = mat4x3( P2,   H,   F,   P1 ) * fw;
	vec3 c2 = mat4x3( P0,   E,   I,   P3 ) * fw;

	/* Smoothly blends the two strongest directions. */
	vec3 color =  d_edge == 0.0 ? 0.5*(c1 + c2) : mix(c1, c2, smoothstep(-limits, limits, d_edge));

	/* Anti-ringing code. */
	vec3 min_sample = min4( E, F, H, I );
	vec3 max_sample = max4( E, F, H, I );
	color = clamp(color, min_sample, max_sample);
	
   FragColor = vec4(color, 1.0);
}
