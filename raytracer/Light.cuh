/*
	Light - Sam Collier
*/
#ifndef LIGHTH
#define LIGHTH

class Light
{
public:
	__device__ virtual Vector3 computeLightDirection(Vector3 surfacePoint);
	__device__ virtual Vector3 computeLightColour(Vector3 surfacePoint);
	__device__ virtual float computeLightDistance(Vector3 surfacePoint);

};

class PointLight : public Light
{
public:
	Vector3 intensity, location;
	__device__ PointLight(Vector3 c, Vector3 pos) : intensity(c), location(pos) {};
	__device__ virtual Vector3 computeLightDirection(Vector3 surfacePoint)
	{
		return unitVector(location - surfacePoint);
	}
	__device__ virtual Vector3 computeLightColour(Vector3 surfacePoint)
	{
		return intensity;
	}
	__device__ virtual float computeLightDistance(Vector3 surfacePoint)
	{
		return (location - surfacePoint).length();
	}
};

#endif