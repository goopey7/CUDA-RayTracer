#ifndef TRANSH
#define TRANSH

#include "Hitable.cuh"

class Translate : public Surface
{
public:
	Vector3 offset;
	Surface* hitablePtr;
	Material* matPtr;
	__device__ Translate(Surface* p,const Vector3 &displacement):hitablePtr(p),offset(displacement),matPtr(p->matPtr){}
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const;
};

__device__ inline bool Translate::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
{
	Ray movedR(r.origin() - offset, r.direction(), r.time());
	if (hitablePtr->hit(movedR, tMin, tMax, rec))
	{
		rec.p += offset;
		return true;
	}
	return false;
}

__device__ inline bool Translate::boundingBox(float t0, float t1, Aabb &box) const
{
	if(hitablePtr->boundingBox(t0,t1,box))
	{
		box = Aabb(box.min() + offset, box.max() + offset);
		return true;
	}
	return false;
}

class RotateY : public Surface
{
public:
	Surface* hitablePtr;
	float sinTheta, cosTheta;
	bool bHasBox;
	Aabb bbox;
	
	__device__ RotateY(Surface* p, float angle);
	__device__ virtual bool hit(const Ray &r, float tMin, float tMax, Intersect &rec)const;
	__device__ virtual bool boundingBox(float t0, float t1, Aabb &box)const
	{
		box = bbox;
		return bHasBox;
	}
};

__device__ RotateY::RotateY(Surface* p, float angle) : hitablePtr(p)
{
	float radians = (M_PI / 180.f) * angle;
	sinTheta = sin(radians);
	cosTheta = cos(radians);
	bHasBox = hitablePtr->boundingBox(0, 1, bbox);
	Vector3 min(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<2;j++)
		{
			for(int k=0;k<2;k++)
			{
				float x = i * bbox.max().x() + (1 - i) * bbox.min().x();
				float y = j * bbox.max().y() + (1 - j) * bbox.min().y();
				float z = k * bbox.max().z() + (1 - k) * bbox.min().z();
				float newX = cosTheta * x + sinTheta * z;
				float newZ = -sinTheta * x + cosTheta * z;
				Vector3 tester(newX, y, newZ);
				for(int c=0;c<3;c++)
				{
					if (tester[c] > max[c])
						max[c] = tester[c];
					if (tester[c] < min[c])
						min[c] = tester[c];
				}
			}
		}
	}
	bbox = Aabb(min, max);
}

__device__ inline bool RotateY::hit(const Ray &r, float tMin, float tMax, Intersect &rec) const
{
	Vector3 origin = r.origin();
	Vector3 direction = r.direction();
	origin[0] = cosTheta * r.origin()[0] - sinTheta * r.origin()[2];
	origin[2] = sinTheta * r.origin()[0] + cosTheta * r.origin()[2];
	direction[0] = cosTheta * r.direction()[0] - sinTheta * r.direction()[2];
	direction[2] = sinTheta * r.direction()[0] + cosTheta * r.direction()[2];
	Ray rotatedR(origin, direction, r.time());
	if(hitablePtr->hit(rotatedR,tMin,tMax,rec))
	{
		Vector3 p = rec.p;
		Vector3 normal = rec.normal;
		p[0] = cosTheta * rec.p[0] + sinTheta * rec.p[2];
		p[2] = -sinTheta * rec.p[0] + cosTheta * rec.p[2];
		normal[0] = cosTheta * rec.normal[0] + sinTheta * rec.normal[2];
		normal[2] = -sinTheta * rec.normal[0] + cosTheta * rec.normal[2];
		rec.p = p;
		rec.setFaceNormal(rotatedR, normal);
		return true;
	}
	return false;
}



#endif