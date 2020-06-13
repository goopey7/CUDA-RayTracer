/*
	Volumes - Sam Collier
*/
#ifndef VOLH
#define VOLH
#include <curand_kernel.h>
#include "Material.cuh"
#include "Surface.cuh"
#include "Texture.cuh"

class ConstantMedium : public Surface
{
public:
	Hitable* boundary;
	float negInvdensity;
	Material* matPtr;
	curandState* randState;
	
	__device__ ConstantMedium(Hitable* b, float d, Texture* a,curandState* localRandState) : boundary(b), negInvdensity(-1/d),randState(localRandState)  { matPtr = new Isotropic(a); }
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0,float t1, Aabb &box)const
	{
		return boundary->boundingBox(t0, t1, box);
	}
};
__device__ inline bool ConstantMedium::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
{
	Intersect rec1, rec2;
	if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
		return false;
	if (!boundary->hit(r, rec1.t + .0001f, FLT_MAX, rec2))
		return false;
	if (rec1.t < tMin)rec1.t = tMin;
	if (rec2.t > tMax)rec2.t = tMax;
	if (rec1.t >= rec2.t)return false;
	if (rec1.t < 0)rec1.t = 0;

	const auto rayLength = r.direction().length();
	const auto distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
	const auto hitDistance = negInvdensity * log(curand_uniform(randState));

	if (hitDistance > distanceInsideBoundary)return false;
	rec.t = rec1.t + hitDistance / rayLength;
	rec.p = r.pointAtParameter(rec.t);
	rec.normal = Vector3(1, 0, 0);
	rec.bFrontFace = true;
	rec.matPtr = this->matPtr;
	return true;
}
#endif
