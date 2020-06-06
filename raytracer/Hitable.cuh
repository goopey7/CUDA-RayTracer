/*
	Intersect, Hitable, Bounding Volumes - Sam Collier
*/
#ifndef HITABLEH
#define HITABLEH
#include "Aabb.cuh"

class Material;

struct Intersect
{
	float t, u, v;
	Vector3 p;
	Vector3 normal;
	Material* matPtr;
	bool bFrontFace;
	__device__ inline void setFaceNormal(const Ray &r,const Vector3 &outwardNormal)
	{
		bFrontFace = dot(r.direction(), outwardNormal) < 0;
		normal = bFrontFace ? outwardNormal : -outwardNormal;
	}
};
class Hitable
{
public:
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec)const = 0;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const = 0;
};

#endif