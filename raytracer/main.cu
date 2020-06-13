/*
	main - Sam Collier
*/
#include <iostream>
#include <time.h>
#include <float.h>
#include <thread>
#include <future>
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "writeImage.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector3.cuh"
#include "Ray.cuh"
#include "HitableList.cuh"
#include "Sphere.cuh"
#include "Triangle.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Surface.cuh"
#include "Rectangle.cuh"
#include "Box.cuh"
#include "Volumes.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include <string>


#include "BvhNode.cuh"
#include "stbImage.h"
#include "Transform.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

//We can't use recursion here, as described in the book, because function calls are valuable. Recursion completely decimated the stack when I tried it
//We can use iteration instead because we have to limit the number of bounces either way.
__device__ Vector3 colour(const Ray &r, Hitable** world, int depth, curandState* localRandState)
{
	Ray currentRay = r;
	Vector3 currentAttenuation = Vector3(1.f, 1.f, 1.f);
	Vector3 currentEmitted = Vector3(0.f, 0.f, 0.f);
	for (int i = 0; i < depth; i++)
	{
		Intersect rec;
		if ((*world)->hit(currentRay, .001f, FLT_MAX, rec))
		{
			Ray scattered;
			Vector3 attenuation;
			Vector3 emitted = rec.matPtr->emitted(rec.u, rec.v, rec.p);
			if (rec.matPtr->scatter(currentRay, rec, attenuation, scattered, localRandState))
			{
				currentAttenuation *= attenuation;
				currentEmitted += emitted * currentAttenuation;
				currentRay = scattered;
			}
			else return currentEmitted + emitted * currentAttenuation;
		}
		else return currentEmitted;
	}
	return currentEmitted; //we have exceeded recursion
}
__device__ Vector3 colourUnlit(const Ray &r, Hitable** world, int depth, curandState* localRandState)
{
	Ray currentRay = r;
	Vector3 currentAttenuation = Vector3(1.f, 1.f, 1.f);
	for (int i = 0; i < depth; i++)
	{
		Intersect rec;
		if ((*world)->hit(currentRay, .001f, FLT_MAX, rec))
		{
			Ray scattered;
			Vector3 attenuation;
			if (rec.matPtr->scatter(currentRay, rec, attenuation, scattered, localRandState))
			{
				currentAttenuation *= attenuation;
				currentRay = scattered;
			}
			else return Vector3(0.f, 0.f, 0.f);
		}
		else
		{
			Vector3 unitDir = unitVector(currentRay.direction());
			float t = .5f * (unitDir.y() + 1.f);
			Vector3 c = (1.f - t) * Vector3(1.f, 1.f, 1.f) + t * Vector3(.5f, .7f, 1.f);
			return currentAttenuation * c;
		}
	}
	return Vector3(0.f, 0.f, 0.f); //we have exceeded bounce limit (which is currently 50)
}

__global__ void randInit(curandState* randState)
{
	if (threadIdx.x == 0  &&blockIdx.x == 0)
		curand_init(419, 0, 0, randState);
}

__global__ void renderInit(int maxX, int maxY, curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY)) return;
	int pixelIndex = j * maxX + i;

	// Each thread in the gpu gets the same seed, and a different sequence number.
	curand_init(419 + pixelIndex, 0, 0, &randState[pixelIndex]);
}

__global__ void textureInit(unsigned char* texData, int width, int height, ImageTexture** tex)
{
	if (threadIdx.x == 0  &&blockIdx.x == 0)
		*tex = new ImageTexture(texData, width, height);
}

__global__ void render(Vector3* fb, int maxX, int maxY,
	int numSamples, Camera** cam, Hitable** world, curandState* randState,int gpuId)
{
	if (gpuId == 0)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i >= .5*maxX || j >= maxY)return; //GPU only needs to do half the image
		int pixelIndex = j * maxX + i; //Only worried about left half
		curandState localRandState = randState[pixelIndex];
		Vector3 outCol(0, 0, 0);
		for (int s = 0; s < numSamples; s++)
		{
			float u = float(i + curand_uniform(&localRandState)) / float(maxX);
			float v = float(j + curand_uniform(&localRandState)) / float(maxY);
			Ray r = (*cam)->generateRay(u, v, &localRandState);
			outCol += colour(r, world, 100, &localRandState);
			//outCol += colourUnlit(r, world, 100, &localRandState);
		}
		randState[pixelIndex] = localRandState;
		outCol /= float(numSamples);
		outCol[0] = sqrtf(outCol[0]);
		outCol[1] = sqrtf(outCol[1]);
		outCol[2] = sqrtf(outCol[2]);
		fb[pixelIndex] = outCol;
	}
	else if(gpuId==1)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i >= .5 * maxX || j >= maxY)return;
		i += .5 * maxX; //GPU 1 Works on the right side
		int pixelIndex = j * maxX + i;
		curandState localRandState = randState[pixelIndex];
		Vector3 outCol(0, 0, 0);
		for (int s = 0; s < numSamples; s++)
		{
			float u = float(i + curand_uniform(&localRandState)) / float(maxX);
			float v = float(j + curand_uniform(&localRandState)) / float(maxY);
			Ray r = (*cam)->generateRay(u, v, &localRandState);
			outCol += colour(r, world, 100, &localRandState);
			//outCol += colourUnlit(r, world, 100, &localRandState);
		}
		randState[pixelIndex] = localRandState;
		outCol /= float(numSamples);
		outCol[0] = sqrtf(outCol[0]);
		outCol[1] = sqrtf(outCol[1]);
		outCol[2] = sqrtf(outCol[2]);
		fb[pixelIndex] = outCol;
	}
	else
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i >= maxX || j >= maxY)return; //Don't want to do waste computer power on unnecessary calculations
		int pixelIndex = j * maxX + i;
		curandState localRandState = randState[pixelIndex];
		Vector3 outCol(0, 0, 0);
		for (int s = 0; s < numSamples; s++)
		{
			float u = float(i + curand_uniform(&localRandState)) / float(maxX);
			float v = float(j + curand_uniform(&localRandState)) / float(maxY);
			Ray r = (*cam)->generateRay(u, v, &localRandState);
			outCol += colour(r, world, 100, &localRandState);
			//outCol += colourUnlit(r, world, 100, &localRandState);
		}
		randState[pixelIndex] = localRandState;
		outCol /= float(numSamples);
		outCol[0] = sqrtf(outCol[0]);
		outCol[1] = sqrtf(outCol[1]);
		outCol[2] = sqrtf(outCol[2]);
		fb[pixelIndex] = outCol;
	}
}

#define RND (curand_uniform(&localRandState))

//Scene 1: Loads Of Spheres!
__device__ inline void scene1(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, ImageTexture** texture, curandState* randState)
{
	curandState localRandState = *randState;
	Texture* checker = new CheckerTexture(new ConstantTexture(Vector3(.2f, .3f, .1f)),
		new ConstantTexture(Vector3(.9f, .9f, .9f)));
	dList[0] = new Sphere(Vector3(0, -1000.0, -1), 1000,
		new Lambert(checker));
	int i = 1;
	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float chooseMat = RND;
			Vector3 centre(a + RND, .2f, b + RND);
			if (chooseMat < .8f)
			{
				dList[i++] = new MovingSphere(centre, centre + Vector3(0, .5f * RND, 0), 0.f, 1.f, .2f, new Lambert(new ConstantTexture(Vector3(RND * RND, RND * RND, RND * RND))));
			}
			else if (chooseMat < .95f)
			{
				dList[i++] = new Sphere(centre, .2f, new Metal(new ConstantTexture(Vector3(.5f * (1.f + RND), .5f * (1.f + RND), .5f * (1.f + RND))), 0.5f * RND));
			}
			else
			{
				dList[i++] = new Sphere(centre, .2f, new Glass(1.5f));
			}
		}
	}
	dList[i++] = new Sphere(Vector3(-4, 1, 0), 1.0, new Glass(1.5f));
	dList[i++] = new Sphere(Vector3(0, 1, 0), 1.f, new Lambert(*texture));
	dList[i++] = new Sphere(Vector3(4, 1, 0), 1.0, new Metal(new ConstantTexture(Vector3(0.7, 0.6, 0.5)), .1));
	*randState = localRandState;
	*dWorld = new HitableList(dList, 22 * 22 + 1 + 3);
	Vector3 lookfrom(13, 2, 3);
	Vector3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.0;
	*dCamera = new Camera(lookfrom,
		lookat,
		Vector3(0, 1, 0),
		20.f,
		float(width) / float(height),
		aperture,
		dist_to_focus,
		0.f, 1.f);
}

//Scene 2: Lighting up a sphere
__device__ inline void scene2(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, ImageTexture** texture, curandState* randState)
{
	curandState localRandState = *randState;
	Texture* checker = new CheckerTexture(new ConstantTexture(Vector3(.2f, .3f, .1f)),
		new ConstantTexture(Vector3(.9f, .9f, .9f)));
	int i = 0;
	dList[i++] = new Sphere(Vector3(0.f, -1000.f, -1.f), 1000, new Lambert(checker));
	//dList[i++] = new Sphere(Vector3(0, 1, 0), 1.f, new Lambert(*texture));
	dList[i++] = new RotateY(new Sphere(Vector3(0, 1, 0), 1, new Lambert(*texture)), 0);
	dList[i++] = new RotateY(new Sphere(Vector3(0, 7, 0), 2, new DiffuseLight(new ConstantTexture(Vector3(4, 4, 4)))), 120);
	//dList[i++] = new XYRect(3, 5, 1, 3, -2, new DiffuseLight(new ConstantTexture(Vector3(4, 4, 4))));
	*dWorld = new HitableList(dList, i);
	Vector3 lookfrom(13, 2, 3);
	Vector3 lookat(0, 0, 0);
	float distToFocus = 10.0;
	float aperture = 0.1;
	*dCamera = new Camera(lookfrom,
		lookat,
		Vector3(0, 1, 0),
		30.0,
		float(width) / float(height),
		aperture,
		distToFocus,
		0.f, 1.f);
}

//Scene 3: Cornell Box
__device__ inline void scene3(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, ImageTexture** texture, curandState* randState)
{
	int i = 0;
	Material* red = new Lambert(new ConstantTexture(Vector3(.65, .05, .05)));
	Material* white = new Lambert(new ConstantTexture(Vector3(.73, .73, .73)));
	Material* green = new Lambert(new ConstantTexture(Vector3(.12, .45, .15)));
	Material* light = new DiffuseLight(new ConstantTexture(Vector3(1, 1, 1)));
	Material* earth = new Lambert(*texture);
	dList[i++] = new YZRect(0, 555, 0, 555, 555, green);
	dList[i++] = new YZRect(0, 555, 0, 555, 0, red);
	dList[i++] = new XZRect(113, 443, 127, 432, 554, light);
	dList[i++] = new XZRect(0, 555, 0, 555, 555, white);
	dList[i++] = new XZRect(0, 555, 0, 555, 0, white);
	dList[i++] = new FlipNormals(new XYRect(0, 555, 0, 555, 555, white));
	Hitable* b1 = new Translate(new RotateY(new Box(Vector3(0, 0, 0), Vector3(165, 165, 165), white), -18), Vector3(130, 0, 65));
	Hitable* b2 = new Translate(new RotateY(new Box(Vector3(0, 0, 0), Vector3(165, 330, 165), white), 15), Vector3(265, 0, 295));
	dList[i++] = b1;
	dList[i++] = b2;
	//dList[i++] = new ConstantMedium(b1, .01, new ConstantTexture(Vector3(1.f, 1.f, 1.f)), randState);
	//dList[i++] = new ConstantMedium(b2, .01, new ConstantTexture(Vector3(0.f, 0.f, 0.f)), randState);
	*dWorld = new HitableList(dList, i);
	Vector3 lookfrom(278, 278, -800);
	Vector3 lookat(278, 278, 0);
	float distToFoucs = 10.0;
	float aperture = 0.0;
	float vFoV = 40.f;
	*dCamera = new Camera(lookfrom, lookat, Vector3(0, 1, 0), vFoV, float(width) / float(height),
		aperture, distToFoucs, 0.f, 1.f);
}

// Scene 4: BvhNode Test
__device__ inline void scene4(Hitable** list,Hitable** world,Camera** dCamera,int width,int height,curandState* randState)
{
	curandState localRandState = *randState;
	int nb = 4;
	Hitable** boxlist1 = new Hitable * [1000];
	Material* ground = new Lambert(new ConstantTexture(Vector3(0.48, 0.83, 0.53)));

	int b = 0;
	for (int i = 0; i < nb; i++) {
		for (int j = 0; j < nb; j++) {
			float w = 100;
			float x0 = -1000 + w * i;
			float z0 = -1000 + w * j;
			float y0 = 0;
			float x1 = x0 + w;
			float y1 = (RND + 0.01) * 100;
			float z1 = z0 + w;
			boxlist1[b++] = new Box(Vector3(x0, y0, z0), Vector3(x1, y1, z1), ground);
		}
	}

	int l = 0;
	printf("whoopsie\n");
	list[l++] = new BvhNode(boxlist1, b, 0, 1, &localRandState);
	list[l++] = new Sphere(Vector3(0, 10, 0), 1, new DiffuseLight(new ConstantTexture(Vector3(8, 8, 8))));
	Texture* checker = new CheckerTexture(new ConstantTexture(Vector3(.2f, .3f, .1f)),
		new ConstantTexture(Vector3(.9f, .9f, .9f)));
	//list[l++] = new Sphere(Vector3(0.f, -1000.f, -1.f), 1000, new Lambert(checker));
	printf("OHHHHHHH\n");
	*world = new HitableList(list, l);
	Vector3 lookfrom(13, 2, 3);
	Vector3 lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.0;
	float vfov = 20.0;
	*dCamera = new Camera(lookfrom,
		lookat,
		Vector3(0, 1, 0),
		vfov,
		float(width) / float(height),
		aperture,
		dist_to_focus,
		0.f, 1.f);
}

//Scene 5: Ray Tracing: The Next Week, Final Scene
__device__ inline void scene5(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, ImageTexture** texture, curandState* randState)
{
	curandState localRandState = *randState;
	float nb = 20;
	Hitable** boxList = new Hitable * [1000];
	Hitable** boxList2 = new Hitable * [1000];
	Material* white = new Lambert(new ConstantTexture(Vector3(.73, .73, .73)));
	Material* ground = new Lambert(new ConstantTexture(Vector3(.48, .83, .53)));

	int b = 0;
	for(float i=0;i < nb;i++)
	{
		for(float j=0;j < nb;j++)
		{

			float w = 150;
			//float c = 100;
			float x0 = -1000+i*w;
			float z0 = -1000+j*w;
			float y0 = 0;
			float x1 = (x0 + w);
			float y1 = 100 * (RND + 0.01);
			float z1 = (z0 + w);
			boxList[b++] = new Box(Vector3(x0, y0, z0), Vector3(x1, y1, z1), ground);
		}
	}
	int l = 0;
	dList[l++] = new BvhNode(boxList, b, 0, 200, randState);
	Material* light = new DiffuseLight(new ConstantTexture(Vector3(3,3,3)));
	dList[l++] = new XZRect(123, 423, 147, 412, 554, light);
	Vector3 centre(400, 400, 200);
	dList[l++] = new MovingSphere(centre, centre + Vector3(30, 0, 0), 0, 1, 50, new Lambert(new ConstantTexture(Vector3(.7, .3, .1))));
	dList[l++] = new Sphere(Vector3(260, 150, 45), 50, new Glass(1.5));
	dList[l++] = new Sphere(Vector3(0, 150, 145), 50, new Metal(new ConstantTexture(Vector3(.8, .8, .9)), 10.f));
	Hitable* boundary = new Sphere(Vector3(360, 150, 145), 70, new Glass(1.5));
	dList[l++] = boundary;
	dList[l++] = new ConstantMedium(boundary, .2, new ConstantTexture(Vector3(.2, .4, .9)),randState);
	boundary = new Sphere(Vector3(0, 0, 0), 5000, new Glass(1.5f));
	dList[l++] = new ConstantMedium(boundary, .0001, new ConstantTexture(Vector3(1, 1, 1)), randState);
	Material* earth = new Lambert(*texture);
	dList[l++] = new Sphere(Vector3(400, 200, 400), 100, earth);
	int ns = 1000;
	for (int j = 0; j < ns; j++)
		boxList2[j] = new Sphere(Vector3(165 * curand_uniform(randState), 165 * curand_uniform(randState), 165 * curand_uniform(randState)), 10, white);
	dList[l++] = new Translate(new RotateY(new BvhNode(boxList2, ns, 0.f, 1.f,randState), 15), Vector3(-100, 270, 395));
	*dWorld = new HitableList(dList, l);
	Vector3 lookfrom(478, 278, -600);
	Vector3 lookat(278, 278, 0);
	float distToFoucs = 10.f;
	float aperture = 0.f;
	float vfov = 40.f;
	*dCamera = new Camera(lookfrom,
		lookat,
		Vector3(0, 1, 0),
		vfov,
		float(width) / float(height),
		aperture,
		distToFoucs,
		0.f, 1.f);
}

//Scene 6: Cornell Box Based Scene
__device__ inline void scene6(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, ImageTexture** texture, curandState* randState)
{
	int i = 0;
	Material* red = new Lambert(new ConstantTexture(Vector3(.65, .05, .05)));
	Material* white = new Lambert(new ConstantTexture(Vector3(.73, .73, .73)));
	Material* green = new Lambert(new ConstantTexture(Vector3(.12, .45, .15)));
	Material* light = new DiffuseLight(new ConstantTexture(Vector3(1, 1, 1)));
	Material* glass = new Glass(1.5);
	Material* earth = new Lambert(*texture);
	dList[i++] = new YZRect(0, 555, 0, 555, 555, green);
	dList[i++] = new YZRect(0, 555, 0, 555, 0, red);
	dList[i++] = new XZRect(113, 443, 127, 432, 554, light);
	dList[i++] = new XZRect(0, 555, 0, 555, 555, white);
	dList[i++] = new XZRect(0, 555, 0, 555, 0, white);
	dList[i++] = new FlipNormals(new XYRect(0, 555, 0, 555, 555, white));
	Hitable* b1 = new Translate(new RotateY(new Box(Vector3(0, 0, 0), Vector3(165, 165, 165), glass), -18), Vector3(130, 0, 65));
	//Hitable* b2 = new Translate(new RotateY(new Box(Vector3(0, 0, 0), Vector3(165, 330, 165), white), 15), Vector3(265, 0, 295));
	dList[i++] = b1;

	Hitable** boxList = new Hitable*[1000];
	int ns = 1000; //num spheres in bounding box
	for (int j = 0; j < ns; j++)
		boxList[j] = new Sphere(Vector3(165 * curand_uniform(randState), 330 * curand_uniform(randState), 165 * curand_uniform(randState)), 10, white);
	dList[i++] = new Translate(new RotateY(new BvhNode(boxList, ns, 0.f, 1.f, randState), 15), Vector3(265, 0, 295));
	//dList[i++] = new ConstantMedium(b1, .01, new ConstantTexture(Vector3(1.f, 1.f, 1.f)), randState);
	//dList[i++] = new ConstantMedium(b2, .01, new ConstantTexture(Vector3(0.f, 0.f, 0.f)), randState);
	*dWorld = new HitableList(dList, i);
	Vector3 lookfrom(278, 278, -800);
	Vector3 lookat(278, 278, 0);
	float distToFoucs = 10.0;
	float aperture = 0.0;
	float vFoV = 40.f;
	*dCamera = new Camera(lookfrom, lookat, Vector3(0, 1, 0), vFoV, float(width) / float(height),
		aperture, distToFoucs, 0.f, 1.f);
}

//Scene 7: Final Scene
__device__ inline void scene7(Hitable** dList,Hitable** dWorld,Camera** dCamera,int width,int height,ImageTexture** texture,curandState* randState)
{
	curandState localRandState = *randState;
	float nb = 20;
	Hitable** boxList = new Hitable * [1000];
	//Hitable** starList = new Hitable * [2000];
	//Hitable** starList2 = new Hitable * [2000];
	Hitable** boxList2 = new Hitable * [1000];
	Material* white = new Lambert(new ConstantTexture(Vector3(.73, .73, .73)));
	Material* ground = new Lambert(new ConstantTexture(Vector3(.48, .83, .53)));
	//Material* starLight = new DiffuseLight(new ConstantTexture(Vector3(7, 7, 7)));
	Material* light = new DiffuseLight(new ConstantTexture(Vector3(1, 1, 1)));
	Material* greenLight = new DiffuseLight(new ConstantTexture(Vector3(2*(22/255.f), 2*(217/255.f), 2*(25/255.f))));

	int b = 0;
	for (float i = 0; i < nb; i++)
	{
		for (float j = 0; j < nb; j++)
		{
			float w = 100*(RND+.01f);
			//float c = 100;
			float x0 = -1000 + i * w;
			float z0 = -1000 + j * w;
			float y0 = 0;
			float x1 = (x0 + w);
			float y1 = 100 * (RND + 0.01f);
			float z1 = (z0 + w);
			boxList[b++] = new Box(Vector3(x0, y0, z0), Vector3(x1, y1, z1), ground);
		}
	}
	/*for (int i = 0; i < 2000; i++)
	{
		starList[i] = new Sphere(Vector3(1000 * curand_uniform(randState), 10 * curand_uniform(randState), 1000 * curand_uniform(randState)), 0.5f, starLight);
		//starList2[i] = new Sphere(Vector3(1000 * curand_uniform(randState), 10 * curand_uniform(randState), 1000 * curand_uniform(randState)), 0.5f, starLight);
	}*/
	int l = 0;
	dList[l++] = new BvhNode(boxList, b, 0, 200, randState);
	dList[l++] = new XZRect(-1000, 1000, -1000, 1000, -3000, greenLight);
	Vector3 centre(400, 400, 200);
	dList[l++] = new MovingSphere(centre, centre + Vector3(30, 0, 0), 0, 1, 50, new Lambert(new ConstantTexture(Vector3(.7, .3, .1))));
	dList[l++] = new Sphere(Vector3(260, 150, 45), 50, new Glass(1.5));
	dList[l++] = new Sphere(Vector3(0, 150, 145), 50, new Metal(new ConstantTexture(Vector3(.8, .8, .9)), 10.f));
	Hitable* boundary = new Sphere(Vector3(360, 150, 145), 70, new Glass(1.5));
	dList[l++] = boundary;
	dList[l++] = new ConstantMedium(boundary, .2, new ConstantTexture(Vector3(.2, .4, .9)), randState);
	boundary = new Sphere(Vector3(0, 0, 0), 5000, new Glass(1.5f));
	dList[l++] = new ConstantMedium(boundary, .0001, new ConstantTexture(Vector3(1, 1, 1)), randState);
	Material* earth = new Lambert(*texture);
	dList[l++] = new Sphere(Vector3(400, 200, 400), 100, earth);
	int ns = 1000;
	for (int j = 0; j < ns; j++)
		boxList2[j] = new Sphere(Vector3(165 * curand_uniform(randState), 165 * curand_uniform(randState), 165 * curand_uniform(randState)), 10, white);
	dList[l++] = new Translate(new RotateY(new BvhNode(boxList2, ns, 0.f, 1.f, randState), 15), Vector3(-100, 270, 395));
	//dList[l++] = new Translate(new BvhNode(starList, 1000, 0.f, 1.f, randState), Vector3(-400, 550, -500));
	//dList[l++] = new Translate(new BvhNode(starList2, 1000, 0.f, 1.f, randState), Vector3(-400, 300, -500));
	dList[l++] = new Sphere(Vector3(0, 1500, 0), 500, light);
	*dWorld = new HitableList(dList, l);
	Vector3 lookfrom(478, 278, -600);
	Vector3 lookat(278, 278, 0);
	float distToFoucs = 10.f;
	float aperture = 0.f;
	float vfov = 40.f;
	*dCamera = new Camera(lookfrom,
		lookat,
		Vector3(0, 1, 0),
		vfov,
		float(width) / float(height),
		aperture,
		distToFoucs,
		0.f, 1.f);
}

//Select active scene here
__global__ void createWorld(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, ImageTexture** texture, curandState* randState)
{
	if (threadIdx.x == 0  &&blockIdx.x == 0)
	{
		//scene1(dList, dWorld, dCamera, width, height, texture, randState);
		//scene2(dList, dWorld, dCamera, width, height, texture, randState);
		scene3(dList, dWorld, dCamera, width, height, texture, randState);
		//scene4(dList, dWorld, dCamera, width, height, randState);
		//scene5(dList, dWorld, dCamera, width, height, texture, randState);
		//scene6(dList, dWorld, dCamera, width, height, texture, randState);
		//scene7(dList, dWorld, dCamera, width, height, texture, randState);
	}
}

__global__ void freeWorld(Hitable** dList, Hitable** dWorld, Camera** dCamera, int numObjects)
{
	for (int i = 0; i < numObjects; i++)
	{
		if((Surface*)dList[i]!=nullptr)
			delete ((Surface*)dList[i])->matPtr;
		delete dList[i];
	}
	delete* dWorld;
	delete* dCamera;
}

Vector3* gpuRender(int width,int height,int numSamples,int tx,int ty,int gpuId)
{
	if (gpuId != 0  &&gpuId != 1)cudaSetDevice(0);
	else cudaSetDevice(gpuId);
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 100000)); //Bounding boxes require recursion, but this seems to be enough for that
	const int res = width * height;
	size_t fbSize = res * sizeof(Vector3);
	//Allocate frame buffer
	Vector3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fbSize));

	//Initialise and allocate textures
	int texX, texY, texN;
	unsigned char* texDataHost = stbi_load("assets/earthmap.jpg", &texX, &texY, &texN, 0);

	unsigned char* texData;
	checkCudaErrors(cudaMallocManaged(&texData, texX * texY * texN * sizeof(unsigned char)));
	checkCudaErrors(cudaMemcpy(texData, texDataHost, texX * texY * texN * sizeof(unsigned char), cudaMemcpyHostToDevice));

	ImageTexture** texture;
	checkCudaErrors(cudaMalloc((void**)&texture, sizeof(ImageTexture*)));
	textureInit << <1, 1 >> > (texData, texX, texY, texture);

	//Allocate random state
	curandState* dRandState;
	checkCudaErrors(cudaMalloc((void**)&dRandState, res * sizeof(curandState)));
	curandState* dRandState2;
	checkCudaErrors(cudaMalloc((void**)&dRandState2, sizeof(curandState)));

	// we need a 2nd random state to be initialised for the world creation
	randInit << <1, 1 >> > (dRandState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create our world and Camera
	Hitable** dList;
	//cornell box has 8
	int numObjects = 488;
	checkCudaErrors(cudaMalloc((void**)&dList, numObjects * sizeof(Hitable*)));
	Hitable** dWorld;
	checkCudaErrors(cudaMalloc((void**)&dWorld, sizeof(Hitable*)));
	Camera** dCamera;
	checkCudaErrors(cudaMalloc((void**)&dCamera, sizeof(Camera*)));
	createWorld << <1, 1 >> > (dList, dWorld, dCamera, width, height, texture, dRandState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//Render the frame buffer
	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);
	renderInit << <blocks, threads >> > (width, height, dRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	render << <blocks, threads >> > (fb, width, height, numSamples, dCamera, dWorld, dRandState, gpuId);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	freeWorld << <1, 1 >> > (dList, dWorld, dCamera, numObjects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dCamera));
	checkCudaErrors(cudaFree(dRandState));
	checkCudaErrors(cudaFree(dList));
	checkCudaErrors(cudaFree(dWorld));
	checkCudaErrors(cudaFree(texData));

	return fb;
}

Vector3* renderCPU(int width,int height,int numSamples,int tx,int ty)
{
	return nullptr;
}

int main()
{
	// 8k is 7680x4320
	std::cout << "Width: ";
	int width;
	std::cin >> width;
	std::cout << "Height: ";
	int height;
	std::cin >> height;
	std::cout << "Number of Samples: ";
	int numSamples;
	std::cin >> numSamples;
	std::cout << "# of GPUs: ";
	int numGPUs;
	std::cin >> numGPUs;
	bool bTwoGPUs = numGPUs == 2;
	// Use 24x24 for release
	const int tx = 24;
	const int ty = 24;
	std::cerr << "Rendering a " << width << "x" << height << " image with "<<(bTwoGPUs?"2 GPUs":"1 GPU");
	std::cerr << "\nUsing " << tx << "x" << ty << " blocks\n";
	time_t now = time(0);
	char* dt = ctime(&now);
	std::cerr << "Render initiated on " << dt << std::endl;
	clock_t start, stop;
	if (bTwoGPUs)
	{
		start = clock();
		std::future<Vector3*> t0 = std::async(&gpuRender, width, height, numSamples, tx, ty,0);
		std::future<Vector3*> t1 = std::async(&gpuRender, width, height, numSamples, tx, ty,1);
		t0.wait();
		t1.wait();
		stop = clock();
		auto* fb0 = t0.get();
		auto* fb1 = t1.get();
		const auto elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		std::string fileName = "output/multiGPUTesting";
		fileName += ".png";
		std::cerr << "\nRendered in " << elapsed << " seconds to " << fileName << std::endl;

		//Output frame buffer as an image
		uint8_t* pixels = new uint8_t[width * height * 3];
		int index = 0;
		for (int j = height - 1; j >= 0; j--)
		{
			for (int i = 0; i < width; i++)
			{
				size_t pixelIndex = j * width + i;
				float r, g, b;
				if (i < width / 2)
				{
					r = fb0[pixelIndex].r();
					g = fb0[pixelIndex].g();
					b = fb0[pixelIndex].b();
				}
				else
				{
					r = fb1[pixelIndex].r();
					g = fb1[pixelIndex].g();
					b = fb1[pixelIndex].b();
				}
				int ir = int(255.99 * r);
				int ig = int(255.99 * g);
				int ib = int(255.99 * b);

				pixels[index++] = ir;
				pixels[index++] = ig;
				pixels[index++] = ib;
			}
		}
		stbi_write_png(fileName.c_str(), width, height, 3, pixels, width * 3);
		checkCudaErrors(cudaFree(fb0));
		checkCudaErrors(cudaFree(fb1));
	}
	else
	{
		start = clock();
		auto fb = gpuRender(width, height, numSamples, tx, ty,2);
		stop = clock();
		double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
		std::string fileName = "output/singleGPUTesting";
		fileName += ".png";
		std::cerr << "\nRendered in " << elapsed << " seconds to " << fileName << std::endl;
		//Output frame buffer as an image
		uint8_t* pixels = new uint8_t[width * height * 3];
		int index = 0;
		for (int j = height - 1; j >= 0; j--)
		{
			for (int i = 0; i < width; i++)
			{
				size_t pixelIndex = j * width + i;
				float r = fb[pixelIndex].r();
				float g = fb[pixelIndex].g();
				float b = fb[pixelIndex].b();
				int ir = int(255.99 * r);
				int ig = int(255.99 * g);
				int ib = int(255.99 * b);

				pixels[index++] = ir;
				pixels[index++] = ig;
				pixels[index++] = ib;
			}
		}
		stbi_write_png(fileName.c_str(), width, height, 3, pixels, width * 3);
	}
	return 0;
}