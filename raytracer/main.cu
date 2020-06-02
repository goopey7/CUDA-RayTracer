/*
	main - Sam Collier
*/
#include <iostream>
#include <time.h>
#include <float.h>
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "writeImage.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stbImage.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector3.cuh"
#include "Ray.cuh"
#include "HitableList.cuh"
#include "Sphere.cuh"
#include "Triangle.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Rectangle.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
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

//We can't use recursion here. I've tried to, but it completely decimates the stack.
__device__ Vector3 colour(const Ray &r,Hitable** world,int depth,curandState* localRandState)
{
	Ray currentRay = r;
	Vector3 currentAttenuation = Vector3(1.f,1.f,1.f);
	Intersect rec;
	if ((*world)->hit(currentRay, .001f, FLT_MAX, rec))
	{
		Ray scattered;
		Vector3 attenuation;
		Vector3 emitted = rec.matPtr->emitted(rec.u, rec.v, rec.p);
		if (depth<15&&rec.matPtr->scatter(currentRay, rec, attenuation, scattered, localRandState))
		{
			return emitted + attenuation * colour(scattered, world, depth + 1, localRandState);
		}
		return emitted;
	}
	return Vector3(0, 0, 0);
}

__device__ Vector3 colourUnlit(const Ray& r, Hitable** world, int depth, curandState* localRandState)
{
	Ray currentRay = r;
	Vector3 currentAttenuation = Vector3(1.f, 1.f, 1.f);
	for (int i = 0; i < 50; i++)
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
	if (threadIdx.x == 0&&blockIdx.x == 0) 
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

__global__ void render(Vector3* fb,int width,int height,
	int numSamples,Camera** cam, Hitable** world, curandState* randState)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=threadIdx.y+blockIdx.y*blockDim.y;
	if(i>=width||j>=height)return; //Don't want to do waste computer power on unnecessary calculations
	int pixelIndex=j*width+i;
	curandState localRandState = randState[pixelIndex];
	Vector3 outCol(0, 0, 0);
	for (int s = 0; s < numSamples; s++)
	{
		float u = float(i+curand_uniform(&localRandState)) / float(height);
		float v = float(j+curand_uniform(&localRandState)) / float(height);
		Ray r = (*cam)->generateRay(u, v, &localRandState);
		outCol += colourUnlit(r, world,0, &localRandState);
	}
	randState[pixelIndex] = localRandState;
	outCol /= float(numSamples);
	outCol[0] = sqrtf(outCol[0]);
	outCol[1] = sqrtf(outCol[1]);
	outCol[2] = sqrtf(outCol[2]);
	fb[pixelIndex] = outCol;
}

#define RND (curand_uniform(&localRandState))

//Scene 1: Loads Of Spheres!
__device__ inline void scene1(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, curandState* randState)
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
				dList[i++] = new MovingSphere(centre,centre+Vector3(0,.5f*RND,0),0.f,1.f,.2f, new Lambert(new ConstantTexture(Vector3(RND * RND, RND * RND, RND * RND))));
			}
			else if (chooseMat < .95f)
			{
				dList[i++] = new Sphere(centre, .2f, new Metal(new ConstantTexture(Vector3(.5f * (1.f + RND), .5f * (1.f + RND), .5f * (1.f + RND))), 0.5f * RND));
			}
			else
			{
				dList[i++] = new Sphere(centre, .2f, new Dielectric(1.5f));
			}
		}
	}
	dList[i++] = new Sphere(Vector3(0, 1, 0), 1.0, new Dielectric(.2f));
	dList[i++] = new MovingSphere(Vector3(0,1,0), Vector3(1,1,0),0.f,1.f,1.f,new Lambert(new ConstantTexture(Vector3(1.f,0.f,0.f))));
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
		0.f,1.f);
}

__device__ inline void scene2(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, curandState* randState)
{
	curandState localRandState = *randState;
	Texture* checker = new CheckerTexture(new ConstantTexture(Vector3(.2f, .3f, .1f)),
		new ConstantTexture(Vector3(.9f, .9f, .9f)));
	//Texture* earth = new ImageTexture(texData, nx, ny);
	dList[0] = new Sphere(Vector3(0, -1000.0, -1), 1000, new Lambert(checker)); //floor
	//dList[1] = new Sphere(Vector3(0, 1, 0), 1, new Lambert(checker));
	dList[1] = new Sphere(Vector3(0, 1, 0), 1, new Lambert(new ConstantTexture(Vector3(0.9,0.9,0.9))));
	dList[2] = new XYRect(1, 2, 1, 2, -2, new DiffuseLight(new ConstantTexture(Vector3(4, 4, 4))));
	*randState = localRandState;
	*dWorld = new HitableList(dList, 3);
	Vector3 lookfrom(7, 3, 7);
	Vector3 lookat(0, 0, 0);
	float dist_to_focus = 10.0; (lookfrom - lookat).length();
	float aperture = 0.1;
	*dCamera = new Camera(lookfrom,
		lookat,
		Vector3(0, 1, 0),
		30.f,
		float(width) / float(height),
		aperture,
		dist_to_focus,
		0.f,1.f);
}

__device__ inline void cornellBox(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, curandState* randState)
{
	curandState localRandState = *randState;
	int i = 0;
	Material* red = new Lambert(new ConstantTexture(Vector3(.65f, .05f, .05f)));
	Material* white = new Lambert(new ConstantTexture(Vector3(.73f, .73f, .73f)));
	Material* green = new Lambert(new ConstantTexture(Vector3(.12f, .45f, .15f)));
	Material* light = new DiffuseLight(new ConstantTexture(Vector3(15.f, 15.f, 15.f)));
	dList[i++] = new YZRect(0.f, 555.f, 0.f, 555.f, 555.f, green);
	dList[i++] = new YZRect(0.f, 555.f, 0.f, 555.f, 0.f, red);
	dList[i++] = new XZRect(213.f, 343.f, 227.f, 332.f, 554.f, light);
	dList[i++] = new XZRect(0.f, 555.f, 0.f, 555.f, 0.f, white);
	dList[i++] = new XZRect(0.f, 555.f, 0.f, 555.f, 555.f, white);
	dList[i++] = new FlipNormals(new XYRect(0, 555, 0, 555, 555, white));
	*randState = localRandState;
	*dWorld = new HitableList(dList, 6);
	Vector3 lookFrom(278, 278, -800);
	Vector3 lookAt(278, 278, 0);
	float distToFocus = 10.f;
	float aperture = 0.f;
	float vFoV = 40.f;
	*dCamera = new Camera(lookFrom, lookAt, Vector3(0, 1, 0), vFoV, float(width) / float(height),
		aperture, distToFocus, 0.f, 1.f);
}

//Select active scene here
__global__ void createWorld(Hitable** dList, Hitable** dWorld,Camera** dCamera,int width,int height,curandState* randState)
{
	if (threadIdx.x == 0&&blockIdx.x == 0)
	{
		//scene1(dList, dWorld, dCamera, width, height, randState);
		scene2(dList, dWorld, dCamera, width, height, randState);
		//cornellBox(dList, dWorld, dCamera, width, height, randState);
	}
}

__global__ void freeWorld(Hitable** dList, Hitable** dWorld,Camera** dCamera,int numObjects)
{
	for (int i = 0; i < numObjects; i++)
	{
		delete ((Surface*)dList[i])->matPtr;
		delete dList[i];
	}
	delete *dWorld;
	delete *dCamera;
}

int main()
{
	// 8k is 7680x4320
	const int width = 800;
	const int height = 600;
	const int numSamples = 10;
	int tx=8;
	int ty=8;
	std::cerr<<"Rendering a "<<width<<"x"<<height<<" image";
	std::cerr<<"\nUsing "<<tx<<"x"<<ty<<" blocks\n";
	int res=width*height;
	size_t fbSize=res*sizeof(Vector3);

	//Allocate frame buffer
	Vector3* fb;
	checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fb),fbSize));

	//Allocate random state
	curandState* dRandState;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dRandState), res * sizeof(curandState)));
	curandState* dRandState2;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dRandState2), res * sizeof(curandState)));

	// we need a 2nd random state to be initialised for the world creation
	randInit<<<1, 1>>>(dRandState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create our world and Camera
	Hitable** dList;
	// 488 objects in scene1
	int numObjects = 3;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dList), numObjects * sizeof(Hitable*)));
	Hitable** dWorld;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dWorld), sizeof(Hitable*)));
	Camera** dCamera;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dCamera), sizeof(Camera*)));
	createWorld<<<1,1>>>(dList, dWorld, dCamera,width,height,dRandState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start,stop;
	

	//Render the frame buffer
	dim3 blocks(width/tx+1,height/ty+1);
	dim3 threads(tx,ty);
	renderInit<<<blocks,threads>>>(width,height,dRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	start = clock();
	render<<<blocks,threads>>>(fb,width,height,numSamples,dCamera,dWorld,dRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop=clock();
	double elapsed=((double)(stop-start))/CLOCKS_PER_SEC;
	std::cerr<<"\nRendered in "<<elapsed<<" seconds.\n";

	//Output frame buffer as an image
	uint8_t* pixels = new uint8_t[width*height*3];
	int index=0;
	for(int j=height-1;j>=0;j--)
	{
		for(int i=0;i<width;i++)
		{
			size_t pixelIndex=j*width+i;
			float r=fb[pixelIndex].r();
			float g=fb[pixelIndex].g();
			float b=fb[pixelIndex].b();

			int ir=int(255.99*r);
			int ig=int(255.99*g);
			int ib=int(255.99*b);
			
			pixels[index++]=ir;
			pixels[index++]=ig;
			pixels[index++]=ib;
		}
	}
	stbi_write_png("scene1.png",width,height,3,pixels,width*3);
	
	//clean up
	freeWorld<<<1,1>>>(dList,dWorld,dCamera,numObjects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dCamera));
	checkCudaErrors(cudaFree(dRandState));
	checkCudaErrors(cudaFree(dList));
	checkCudaErrors(cudaFree(dWorld));
	checkCudaErrors(cudaFree(fb));
	checkCudaErrors(cudaFree(dRandState2));
	return 0;
}