/*
	Surface - Sam Collier
*/
#ifndef SURFACEH
#define SURFACEH
#include "Hitable.cuh"

class Surface : public Hitable
{
public:
	Material* matPtr;
};

class FlipNormals : public Surface
{
public:
	Surface* surfacePtr;
	Material* matPtr;
	__device__ FlipNormals(Surface* p) : surfacePtr(p) { this->matPtr = p->matPtr; }
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
	{
		if (surfacePtr->hit(r, tMin, tMax, rec))
		{
			rec.bFrontFace = !rec.bFrontFace;
			return true;
		}
		return false;
	}
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box) const
	{
		return surfacePtr->boundingBox(t0, t1, box);
	}
};

#endif