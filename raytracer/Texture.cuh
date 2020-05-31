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
#endif