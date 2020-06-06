#ifndef RECTH
#define RECTH
#include "Surface.cuh"

class XYRect : public Surface
{
public:
	float x0, x1, y0, y1, k;
	Material* matPtr;
	__device__ XYRect() {}
	__device__ XYRect(float _x0, float _x1, float _y0, float _y1, float _k, Material* mat) :
		x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), matPtr(mat) {};
	__device__ virtual bool hit(const Ray &r, float t0, float t1, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const
	{
		box = Aabb(Vector3(x0, y0, k - .0001f), Vector3(x1, y1, k + .0001f));
		return true;
	}
};

__device__ bool XYRect::hit(const Ray &r, float t0, float t1, Intersect &rec) const
{
	float t = (k - r.origin().z()) / r.direction().z();
	if (t<t0 || t>t1)return false;
	float x = r.origin().x() + t * r.direction().x();
	float y = r.origin().y() + t * r.direction().y();
	if (x<x0 || x>x1 || y<y0 || y>y1)return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	rec.matPtr = matPtr;
	rec.p = r.pointAtParameter(t);
	rec.setFaceNormal(r,Vector3(0.f, 0.f, 1.f));
	return true;
}

class XZRect : public Surface
{
public:
	float x0, x1, z0, z1, k;
	Material* matPtr;
	__device__ XZRect() {}
	__device__ XZRect(float _x0, float _x1, float _z0, float _z1, float _k, Material* mat) :
		x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), matPtr(mat) {};
	__device__ virtual bool hit(const Ray &r, float t0, float t1, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const
	{
		box = Aabb(Vector3(x0, z0, k - .0001f), Vector3(x1, z1, k + .0001f));
		return true;
	}
};
__device__ bool XZRect::hit(const Ray &r, float t0, float t1, Intersect &rec) const
{
	float t = (k - r.origin().y()) / r.direction().y();
	if (t<t0 || t>t1)return false;
	float x = r.origin().x() + t * r.direction().x();
	float z = r.origin().z() + t * r.direction().z();
	if (x<x0 || x>x1 || z<z0 || z>z1)return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	rec.matPtr = matPtr;
	rec.p = r.pointAtParameter(t);
	rec.setFaceNormal(r,Vector3(0.f, 1.f, 0.f));
	return true;
}

class YZRect : public Surface
{
public:
	float y0, y1, z0, z1, k;
	Material* matPtr;
	__device__ YZRect() {}
	__device__ YZRect(float _y0, float _y1, float _z0, float _z1, float _k, Material* mat) :
		y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), matPtr(mat) {};
	__device__ virtual bool hit(const Ray &r, float t0, float t1, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const
	{
		box = Aabb(Vector3(y0, z0, k - .0001f), Vector3(y1, z1, k + .0001f));
		return true;
	}
};
__device__ bool YZRect::hit(const Ray &r, float t0, float t1, Intersect &rec) const
{
	float t = (k - r.origin().x()) / r.direction().x();
	if (t<t0 || t>t1)return false;
	float y = r.origin().y() + t * r.direction().y();
	float z = r.origin().z() + t * r.direction().z();
	if (y<y0 || y>y1 || z<z0 || z>z1)return false;
	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	rec.matPtr = matPtr;
	rec.p = r.pointAtParameter(t);
	rec.setFaceNormal(r, Vector3(1.f, 0.f, 0.f));
	return true;
}

#endif