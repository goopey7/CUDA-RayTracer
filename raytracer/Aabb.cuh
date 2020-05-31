#ifndef AABBH
#define AABBH

#include "ray.cuh"
// We aren't using the built-in ones because this is faster. Since we aren't worrying about NaNs and other possible exceptions.
__device__ inline float ffMin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffMax(float a, float b) { return a > b ? a : b; }

/*
	Refer to chapter 2 --- Ray Tracing: The Next Week by Peter Shirley
	Since the most algorithmically complex task is the intersect method,
	we can use bounding volumes to reduce the amount of intersect calculations we have to make.
	When calculating whether or not a ray hits a bounding box, we don't need any info besides whether or not
	the ray hit it since bounding boxes aren't being rendered.
	If the ray doesn't hit a bounding box, then we know that it won't hit any objects within it.
	So we wouldn't have to call the intersect methods within a volume unless the ray collides with the volume.
*/
class Aabb //Axis-aligned bounding boxes.
{
public:
	Vector3 _min, _max;
	__device__ Aabb() {}
	__device__ Aabb(const Vector3 &a, const Vector3 &b) { _min = a; _max = b; }
	__device__ Vector3 min() { return _min; }
	__device__ Vector3 max() { return _max; }
	__device__ inline bool hit(const Ray &r, float tMin, float tMax) const
	{
		for (int a = 0; a < 3; a++)
		{
			float invD = 1.f / r.direction()[a];
			float t0 = (_min[a] - r.origin()[a]) * invD;
			float t1 = (_max[a] - r.origin()[a]) * invD;
			if (invD < 0.f)
			{
				float temp = t0;
				t0 = t1;
				t1 = temp;
			}
			tMin = ffMax(t0, tMin);
			tMax = ffMin(t1, tMax);
			if (tMax <= tMin)return false;
		}
		return true;
	}
};
__device__ inline Aabb surroundingBox(Aabb box0, Aabb box1)
{
	Vector3 small(ffMin(box0.min().x(), box1.min().x()),
		ffMin(box0.min().y(), box1.min().y()),
		ffMin(box0.min().z(), box1.min().z()));
	Vector3 big(ffMax(box0.max().x(), box1.max().x()),
		ffMax(box0.max().y(), box1.max().y()),
		ffMax(box0.max().z(), box1.max().z()));
	return Aabb(small, big);
}
#endif