/*
	Triangle - Sam Collier
*/
#ifndef TRIANGLEH
#define TRIANGLEH

#include "Hitable.cuh"
#include "Surface.cuh"
#include "Vector3.cuh"

class Triangle : public Surface
{
public:
	__device__ Triangle() {}
	__device__ Triangle(Vector3 ver0, Vector3 ver1,Vector3 ver2, Material* m) : v0(ver0), v1(ver1), v2(ver2), matPtr(m) {};
	__device__ virtual bool hit(const Ray &r, float tmin, float tmax, Intersect &rec) const;
	Vector3 v0, v1, v2;
	Material* matPtr;
};

__device__ bool Triangle::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
{
	Vector3 normal = unitVector(cross(v1 - v0, v2 - v0));
	float d = dot(Vector3(0.f, 0.f, 0.f) - v0, normal);
	Vector3 origin = r.origin();
	double distance = (-dot(origin, normal) + d) / (dot(r.direction(), normal));
	if (distance>FLT_EPSILON)
	{
		Vector3 inter = r.pointAtParameter(distance);
		double a = dot(cross(v1 - v0, inter - v0), normal);
		double b = dot(cross(v2 - v1, inter - v1), normal);
		double c = dot(cross(v0 - v2, inter - v2), normal);
		if (a > 0  &&b > 0  &&c > 0) //Intersect!
		{
			rec.t = distance;
			rec.p = r.pointAtParameter(rec.t);
			rec.matPtr = matPtr;
			if (dot(normal, r.direction()) > 0)
				rec.normal = -normal;
			else
				rec.normal = normal;
			return true;
		}
	}
	return false;
}

#endif
