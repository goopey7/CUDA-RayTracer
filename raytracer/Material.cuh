/*
	Material - Sam Collier
*/
#ifndef MaterialH
#define MaterialH

struct Intersect;
#include "Ray.cuh"
#include "Hitable.cuh"
#include "Texture.cuh"

__device__ float schlick(float cosine, float refIdx)
{
	float r0 = (1.f - refIdx) / (1.f + refIdx);
	r0 = r0 * r0;
	return r0 + (1.f - r0) * pow((1.f - cosine), 5.f);
}

__device__ bool refract(const Vector3 &v, const Vector3 &n, float niOverNt, Vector3 &refracted)
{
	Vector3 uv = unitVector(v);
	float dt = dot(uv, n);
	float discrim = 1.f - niOverNt * niOverNt * (1 - dt * dt);
	if (discrim > 0)
	{
		refracted = niOverNt * (uv - n * dt) - n * sqrtf(discrim);
		return true;
	}
	return false;
}

#define RANDVector3 Vector3(curand_uniform(localRandState),curand_uniform(localRandState),curand_uniform(localRandState))

__device__ Vector3 randomInUnitSphere(curandState* localRandState)
{
	Vector3 p;
	do p = 2.f * RANDVector3 - Vector3(1.f, 1.f, 1.f);
	while (p.squaredLength() >= 1.f);
	return p;
}

__device__ Vector3 reflect(const Vector3 &v, const Vector3 &n) { return v - 2.f * dot(v, n) * n; }

class Material
{
public:
	__device__ virtual bool scatter(const Ray &rIn, const Intersect &rec, Vector3 &attenuation, Ray &scattered, curandState* localRandState)const = 0;
};

class Lambert : public Material
{
public:
	Texture* albedo;
	__device__ Lambert(Texture* a) : albedo(a) {}
	__device__ virtual bool scatter(const Ray &rIn, const Intersect &rec, Vector3 &attenuation, Ray &scattered, curandState* localRandState) const
	{
		Vector3 target = rec.p + rec.normal + randomInUnitSphere(localRandState);
		scattered = Ray(rec.p, target - rec.p,rIn.time());
		attenuation = albedo->value(0,0,rec.p);
		return true;
	}
};

class Metal : public Material
{
public:
	__device__ Metal(Texture* a, float f)
	{ 
		albedo = a;
		if (f < 1)fuzz = f; else fuzz = 1; 
	}
	__device__ virtual bool scatter(const Ray &rIn, const Intersect &rec, Vector3 &attenuation, Ray &scattered, curandState* localRandState) const
	{
		Vector3 reflected = reflect(unitVector(rIn.direction()), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere(localRandState),rIn.time());
		attenuation = albedo->value(0,0,rec.p);
		return (dot(scattered.direction(), rec.normal) > 0.f);
	}
	Texture* albedo;
	float fuzz;
};

class Dielectric : public Material
{
public:
	float refIdx;
	__device__ Dielectric(float ri) : refIdx(ri) {}
	__device__ virtual bool scatter(const Ray &rIn, const Intersect &rec, Vector3 &attenuation, Ray &scattered, curandState* localRandState) const
	{
		Vector3 outwardNorm;
		Vector3 reflected = reflect(rIn.direction(), rec.normal);
		float niOverNt;
		attenuation = Vector3(1.f, 1.f, 1.f);
		Vector3 refracted;
		float reflectProb;
		float cosine;
		if (dot(rIn.direction(), rec.normal) > 0.f)
		{
			outwardNorm = -rec.normal;
			niOverNt = refIdx;
			cosine = dot(rIn.direction(), rec.normal) / rIn.direction().length();
			cosine = sqrtf(1.f - refIdx * refIdx * (1 - cosine * cosine));
		}
		else
		{
			outwardNorm = rec.normal;
			niOverNt = 1.f / refIdx;
			cosine = -dot(rIn.direction(), rec.normal) / rIn.direction().length();
		}
		if (refract(rIn.direction(), outwardNorm, niOverNt, refracted))
			reflectProb = schlick(cosine, refIdx);
		else reflectProb = 1.f;
		if (curand_uniform(localRandState) < reflectProb)
			scattered = Ray(rec.p, reflected,rIn.time());
		else scattered = Ray(rec.p, refracted,rIn.time());
		return true;
	}
};

#endif