#ifndef BVHNODEH
#define BVHNODEH
#include "Aabb.cuh"
#include "Hitable.cuh"

class BvhNode : public Hitable
{
public:
	__device__ BvhNode() {}
	__device__ BvhNode(Hitable** l, int n, float time0, float time1, curandState* localRandState);
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const;
	Hitable* left, * right;
	Aabb box;
};

bool BvhNode::boundingBox(float t0, float t1, Aabb &b)const
{
	b = box;
	return true;
}

bool BvhNode::hit(const Ray &r, float tMin, float tMax, Intersect &rec)const
{
	if (box.hit(r, tMin, tMax))
	{
		Intersect leftRec, rightRec;
		bool hitLeft = left->hit(r, tMin, tMax, leftRec);
		bool hitRight = right->hit(r, tMin, tMax, rightRec);
		if (hitLeft  && hitRight)
		{
			if (leftRec.t < rightRec.t)
				rec = leftRec;
			else rec = rightRec;
			return true;
		}
		else if (hitLeft)
		{
			rec = leftRec;
			return true;
		}
		else if (hitRight)
		{
			rec = rightRec;
			return true;
		}
		else return false;
	}
	return false;
}
__device__ int boxXCompare(const void* a, const void* b)
{
	Aabb boxLeft, boxRight;
	Hitable* ah = *(Hitable**)a;
	Hitable* bh = *(Hitable**)b;
	if (!ah->boundingBox(0, 0, boxLeft) || !bh->boundingBox(0, 0, boxRight))
		std::cerr << "no bounding box in BvhNode constructor\n";
	if (boxLeft.min().x() - boxRight.min().x() < 0.f) return -1;
	else return 1;
}
__device__ int boxYCompare(const void* a, const void* b)
{
	Aabb boxLeft, boxRight;
	Hitable* ah = *(Hitable**)a;
	Hitable* bh = *(Hitable**)b;
	if (!ah->boundingBox(0, 0, boxLeft) || !bh->boundingBox(0, 0, boxRight))
		std::cerr << "no bounding box in BvhNode constructor\n";
	if (boxLeft.min().y() - boxRight.min().y() < 0.f) return -1;
	else return 1;
}
__device__ int boxZCompare(const void* a, const void* b)
{
	Aabb boxLeft, boxRight;
	Hitable* ah = *(Hitable**)a;
	Hitable* bh = *(Hitable**)b;
	if (!ah->boundingBox(0, 0, boxLeft) || !bh->boundingBox(0, 0, boxRight))
		std::cerr << "no bounding box in BvhNode constructor\n";
	if (boxLeft.min().z() - boxRight.min().z() < 0.f) return -1;
	else return 1;
}
BvhNode::BvhNode(Hitable** l, int n, float time0, float time1, curandState* localRandState)
{
	int axis = int(3 * curand_uniform(localRandState));
	if (axis == 0)
		thrust::sort(1, 1 + n, boxXCompare);
	else if (axis == 1)
		thrust::sort(1, 1 + n, boxYCompare);
	else thrust::sort(1, 1 + n, boxZCompare);
	if (n == 1)left = right = l[0];
	else if (n == 2)
	{
		left = l[0];
		right = l[1];
	}
	else
	{
		left = new BvhNode(1, n / 2, time0, time1);
		right = new BvhNode(1 + n / 2, n - n / 2, time0, time1);
	}
	Aabb boxLeft, boxRight;
	if (!left->boundingBox(time0, time1, boxLeft) || !right->boundingBox(time0, time1, boxRight))
		std::cerr << "no bounding box in BvhNode constructor\n";
	box = surroundingBox(boxLeft, boxRight);
}
#endif