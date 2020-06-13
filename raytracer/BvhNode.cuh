/*
    BvhNode - Sam Collier
*/
#ifndef BVHNODEH
#define BVHNODEH
#include "Aabb.cuh"
#include "Hitable.cuh"
#include <thrust/sort.h>

struct BoxCompare {
	__device__ BoxCompare(int m) : mode(m) {}
	__device__ bool operator()(Hitable* a, Hitable* b) const
	{
		Aabb boxLeft, boxRight;
		Hitable* ah = a;
		Hitable* bh = b;

		if (!ah->boundingBox(0, 0, boxLeft) || !bh->boundingBox(0, 0, boxRight))
        {
			return false;
		}

		float val1, val2;
		if (mode == 1) 
        {
			val1 = boxLeft.min().x();
			val2 = boxRight.min().x();
		}
		else if (mode == 2) 
        {
			val1 = boxLeft.min().y();
			val2 = boxRight.min().y();
		}
		else if (mode == 3) 
        {
			val1 = boxLeft.min().z();
			val2 = boxRight.min().z();
        }

		if (val1 - val2 < 0.0) 
            return false;
		return true;
	}
	// mode: 1, x; 2, y; 3, z
	int mode;
};

class BvhNode : public Hitable
{
public:
	__device__ BvhNode() {}
	__device__ BvhNode(Hitable** l, int n, float time0, float time1, curandState* localRandState);
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const;
	Hitable* left, * right;
	Aabb box;
	//Material* matPtr = nullptr;
};

__device__ inline BvhNode::BvhNode(Hitable** l,int n,float time0,float time1,curandState* state)
{
    int axis = int(3 * curand_uniform(state));
    if (axis == 0)
        thrust::sort(l, l + n, BoxCompare(1));
    else if (axis == 1)
        thrust::sort(l, l + n, BoxCompare(2));
    else
        thrust::sort(l, l + n, BoxCompare(3));
    if (n == 1)
        left = right = l[0];
    else if (n == 2)
    {
        left = l[0];
        right = l[1];
    }
    else
    {
        //printf("n value: %i\n",n);
        left = new BvhNode(l, n / 2, time0, time1, state);
        right = new BvhNode(l + n / 2, n - n / 2, time0, time1, state);
    }
    Aabb boxLeft, boxRight;
    if (!left->boundingBox(time0, time1, boxLeft) ||
        !right->boundingBox(time0, time1, boxRight))
    {
        //No bounding box in BvhNode constructor!";
        return;
    }
    box = surroundingBox(boxLeft, boxRight);
}


__device__ bool BvhNode::boundingBox(float t0,float t1,Aabb &b) const
{
	b = box;
    return true;
}


__device__ bool BvhNode::hit(const Ray &r,float tMin,float tMax,Intersect &rec) const
{
    if (box.hit(r, tMin, tMax))
    {
        Intersect leftRec, rightRec;
        bool hitLeft = left->hit(r, tMin, tMax, leftRec);
        bool hitRight = right->hit(r, tMin, tMax, rightRec);
        if (hitLeft &&hitRight)
        {
            if (leftRec.t < rightRec.t)
                rec = leftRec;
            else
                rec = rightRec;
            return true;
        }
        if (hitLeft)
        {
            rec = leftRec;
            return true;
        }
        if (hitRight)
        {
            rec = rightRec;
            return true;
        }
        return false;
    }
    return false;
}
#endif