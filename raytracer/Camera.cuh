#ifndef CameraH
#define CameraH
#define _USE_MATH_DEFINES

#include "Ray.cuh"
#include <math.h>
#include <curand_kernel.h>

__device__ Vector3 randomInUnitDisk(curandState* localRandState)
{
	Vector3 p;
	do p = 2.0f * Vector3(curand_uniform(localRandState), curand_uniform(localRandState), 0) - Vector3(1.f, 1.f, 0.f);
	while (dot(p, p) > 01.f);
	return p;
}

class Camera
{
public:
	__device__ Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focusDist,
		float t0,float t1)
	{
		time0 = t0;
		time1 = t1;
		lensRadius = aperture / 2.f;
		float theta = vfov * ((float)M_PI) / 180.f;
		float halfHeight = tan(theta / 2);
		float halfWidth = aspect * halfHeight;
		origin = lookfrom;
		w = unitVector(lookfrom - lookat);
		u = unitVector(cross(vup, w));
		v = cross(w, u);
		lowerLeftCorner = origin - halfWidth * u * focusDist - halfHeight * v * focusDist - w * focusDist;
		horizontal = 2.f * halfWidth * u * focusDist;
		vertical = 2.f * halfHeight * v * focusDist;
	}
	__device__ Ray generateRay(float s, float t, curandState* localRandState)
	{
		Vector3 rd = lensRadius * randomInUnitDisk(localRandState);
		Vector3 offset = u * rd.x() + v * rd.y();
		float time = time0 + curand_uniform(localRandState) * (time1 - time0);
		return Ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset,time);
	}
	Vector3 lowerLeftCorner, horizontal, vertical, origin;
	Vector3 u, v, w;
	float lensRadius,time0,time1;
};

#endif