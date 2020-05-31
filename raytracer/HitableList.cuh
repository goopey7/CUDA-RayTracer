/*
	HitableList - Sam Collier
*/
#ifndef HITABLELISTH
#define HITABLELISTH

#include "Hitable.cuh"

class HitableList : public Hitable
{
public:
	__device__ HitableList() {}
	__device__ HitableList(Hitable** l, int n) { list = l; listSize = n; }
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec) const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box) const;
	Hitable** list;
	int listSize;
};
__device__ bool HitableList::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
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
__device__ bool HitableList::boundingBox(float t0, float t1, Aabb &box)const
{
	if (listSize < 1)return false;
	Aabb tempBox;
	bool firstTrue = list[0]->boundingBox(t0, t1, tempBox);
	if (!firstTrue)return false;
	else box = tempBox;
	for (int i = 1; i < listSize; i++)
	{
		if (list[0]->boundingBox(t0, t1, tempBox))
			box = surroundingBox(box, tempBox);
		else return false;
	}
	return true;
}
#endif
