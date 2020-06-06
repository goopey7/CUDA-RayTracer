#ifndef BOXH
#define BOXH

#include "HitableList.cuh"
#include "Rectangle.cuh"

class Box : public Surface
{
public:
	Vector3 pMin, pMax;
	Hitable* listPtr;
	__device__ Box(){}
	__device__ Box(const Vector3 &p0, const Vector3 &p1, Material* matPtr);
	__device__ virtual bool hit(const Ray &r, float t0, float t1, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0,float t1,Aabb &box)const
	{
		box = Aabb(pMin, pMax);
		return true;
	}
};
__device__ inline Box::Box(const Vector3 &p0, const Vector3 &p1, Material* matPtr)
{
	pMin = p0;
	pMax = p1;
	Hitable** list = new Hitable * [6];
	list[0] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), matPtr);
	list[1] = new FlipNormals(new XYRect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), matPtr));
	list[2] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), matPtr);
	list[3] = new FlipNormals(new XZRect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), matPtr));
	list[4] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), matPtr);
	list[5] = new FlipNormals(new YZRect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), matPtr));
	listPtr = new HitableList(list, 6);
}

inline bool Box::hit(const Ray &r, float t0, float t1, Intersect &rec) const
{
	return listPtr->hit(r, t0, t1, rec);
}


#endif