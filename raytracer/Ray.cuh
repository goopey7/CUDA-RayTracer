/*
	Ray - Sam Collier

	We only use Rays on the GPU,
	so only the __device__ flag is
	necessary.
*/
#ifndef RayH
#define RayH

#include "Vector3.cuh"
class Ray
{
public:
	Vector3 A, B;
	float _time;
	__device__ Ray() {}
	__device__ Ray(const Vector3 &a, const Vector3 &b,float ti=0.f)
	{
		A = a;
		B = b;
		_time = ti; //for motion blur effects
	}
	__device__ Vector3 origin() const { return A; }
	__device__ Vector3 direction() const { return B; }
	__device__ float time() const { return _time; }
	__device__ Vector3 pointAtParameter(float t) const { return A + t * B; }
};
#endif
