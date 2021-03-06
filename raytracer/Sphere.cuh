/*
	Sphere - Sam Collier
*/
#ifndef SPHEREH
#define SPHEREH

#include "Hitable.cuh"
#include "Surface.cuh"
#define _USE_MATH_DEFINES
#include "math.h"

class Sphere : public Surface
{
public:
	__device__ Sphere() {}
	__device__ Sphere(Vector3 cen, float r, Material* m) : centre(cen), radius(r), matPtr(m) {};
	__device__ virtual bool hit(const Ray &r, float tmin, float tmax, Intersect &rec) const;
	__device__ virtual bool boundingBox(float t0,float t1,Aabb &box) const;
	__device__ void getSphereUV(const Vector3 &p, float &u, float &v)const;
	Vector3 centre;
	float radius;
	Material* matPtr;
};

__device__ bool Sphere::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
{
	Vector3 oc = r.origin() - centre;
	float a = dot(r.direction(), r.direction());
	float b = 2.f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discrim = b * b - 4.f*a * c;
	if (discrim > 0)
	{
		float temp = (-b - sqrt(discrim)) / (2*a);
		if (temp<tMax&&temp>tMin)
		{
			rec.t = temp;
			rec.p = r.pointAtParameter(rec.t);
			rec.normal = (rec.p - centre) / radius;
			rec.matPtr = matPtr;
			getSphereUV((rec.p - centre) / radius, rec.u, rec.v);
			return true;
		}
		temp = (-b + sqrt(discrim)) / (2*a);
		if (temp<tMax&&temp>tMin)
		{
			rec.t = temp;
			rec.p = r.pointAtParameter(rec.t);
			rec.normal = (rec.p - centre) / radius;
			rec.matPtr = matPtr;
			getSphereUV((rec.p - centre) / radius, rec.u, rec.v);
			return true;
		}
	}
	return false;
}

__device__ bool Sphere::boundingBox(float t0, float t1, Aabb &box) const
{
	box = Aabb(centre - Vector3(radius, radius, radius), centre + Vector3(radius, radius, radius));
	return true;
}

__device__ inline void Sphere::getSphereUV(const Vector3 &p, float &u, float &v) const
{
	float phi = atan2(p.z(), p.x());
	float theta = asin(p.y());
	u = 1 - (phi + M_PI) / (2 * M_PI);
	v = (theta + M_PI / 2) / M_PI;
}


class MovingSphere : public Sphere
{
public:
	__device__ MovingSphere() {}
	__device__ MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material* m) :centre0(cen0), centre1(cen1), time0(t0),
		time1(t1), radius(r), matPtr(m) {};
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec) const;
	__device__ Vector3 centre(float time) const;
	Vector3 centre0, centre1;
	float time0, time1, radius;
	Material* matPtr;
};

__device__ Vector3 MovingSphere::centre(float time) const
{
	return centre0 + ((time - time0) / (time1 - time0)) * (centre1 - centre0);
}

__device__ bool MovingSphere::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
{
	Vector3 oc = r.origin() - centre(r.time());
	float a = dot(r.direction(), r.direction());
	float b = 2.f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discrim = b * b - 4.f * a * c;
	if (discrim > 0)
	{
		float temp = (-b - sqrt(discrim)) / (2 * a);
		if (temp<tMax&&temp>tMin)
		{
			rec.t = temp;
			rec.p = r.pointAtParameter(rec.t);
			rec.normal = (rec.p - centre(r.time())) / radius;
			rec.matPtr = matPtr;
			return true;
		}
		temp = (-b + sqrt(discrim)) / (2 * a);
		if (temp<tMax&&temp>tMin)
		{
			rec.t = temp;
			rec.p = r.pointAtParameter(rec.t);
			rec.normal = (rec.p - centre(r.time())) / radius;
			rec.matPtr = matPtr;
			return true;
		}
	}
	return false;
}

#endif
