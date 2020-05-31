/*
	HitableList - Sam Collier
*/
#ifndef HITABLELISTH
#define HITABLELISTH

#include "Intersect.cuh"

class HitableList : public Hitable
{
public:
	__device__ HitableList() {}
	__device__ HitableList(Hitable** l, int n) { list = l; listSize = n; }
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, Intersect& rec) const;
	Hitable** list;
	int listSize;
};
__device__ bool HitableList::hit(const Ray& r, float tMin, float tMax, Intersect& rec) const
{
	Intersect tempRec;
	bool bHitAnything = false;
	float closestSoFar = tMax;
	for (int i = 0; i < listSize; i++)
	{
		if (list[i]->hit(r, tMin, closestSoFar, tempRec))
		{
			bHitAnything = true;
			closestSoFar = tempRec.t;
			rec = tempRec;
		}
	}
	return bHitAnything;
}
#endif