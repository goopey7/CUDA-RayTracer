/*
	Intersect - Sam Collier
*/
#ifndef HITABLEH
#define HITABLEH

#include "Ray.cuh"

class Material;

struct Intersect
{
	float t;
	Vector3 p;
	Vector3 normal;
	Material* matPtr;
};
class Hitable
{
public:
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, Intersect& rec)const = 0;
};
#endif