#ifndef TEXTUREH
#define TEXTUREH
#include "Vector3.cuh"
class Texture
{
public:
	__device__ virtual Vector3 value(float u, float v, const Vector3 &p)const = 0;
};

class ConstantTexture : public Texture
{
public:
	Vector3 colour;
	__device__ ConstantTexture() {}
	__device__ ConstantTexture(Vector3 c) :colour(c) {}
	__device__ virtual Vector3 value(float u, float v, const Vector3 &p) const
	{
		return colour;
	}
};

class CheckerTexture : public Texture
{
public:
	Texture* odd, *even;
	__device__ CheckerTexture() {}
	__device__ CheckerTexture(Texture* t0, Texture* t1) : even(t0), odd(t1) {}
	__device__ virtual Vector3 value(float u, float v, const Vector3 &p) const
	{
		float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)return odd->value(u, v, p);
		else return even->value(u, v, p);
	}
};

class ImageTexture : public Texture
{
public:
	unsigned char* data;
	int width, height;
	__device__ ImageTexture(){}
	__device__ ImageTexture(unsigned char* pixels,int A,int B):data(pixels),width(A),height(B){}
	__device__ virtual Vector3 value(float u, float v, const Vector3 &p) const;
};

__device__ inline Vector3 ImageTexture::value(float u, float v, const Vector3 &p) const
{
	int i = u * width;
	int j = (1 - v) * height - .001f;
	if (i < 0)i = 0;
	if (j < 0)j = 0;
	if (i > width - 1)i = width - 1;
	if (j > height - 1)j = height - 1;
	float r = int(data[3 * i + 3 * width * j]) / 255.f;
	float g = int(data[3 * i + 3 * width * j+1]) / 255.f;
	float b = int(data[3 * i + 3 * width * j+2]) / 255.f;
	return Vector3(r, g, b);
}

#endif