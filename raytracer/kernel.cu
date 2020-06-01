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
#include "Light.cuh"
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
		else return emitted;
	}
	else
	{
		return Vector3(0, 0, 0);
	}
}

__device__ Vector3 colourUnlit(const Ray& r, Hitable** world, int depth, curandState* localRandState)
{
	Intersect rec;
	if ((*world)->hit(r, .001f, FLT_MAX, rec))
	{
		Ray scattered;
		Vector3 attenuation;
		if (depth < 15 && rec.matPtr->scatter(r, rec, attenuation, scattered, localRandState))
		{
			return attenuation * colourUnlit(scattered, world, depth + 1, localRandState);
		}
		else return Vector3(0, 0, 0);
	}
	else return Vector3(1.f, 1.f, 1.f);
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

__global__ void render(Vector3* fb,int maxX,int maxY,
	int numSamples,Camera** cam, Hitable** world, curandState* randState)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=threadIdx.y+blockIdx.y*blockDim.y;
	if(i>=maxX||j>=maxY)return; //Don't want to do waste computer power on unnecessary calculations
	int pixelIndex=j*maxX+i;
	curandState localRandState = randState[pixelIndex];
	Vector3 outCol(0, 0, 0);
	for (int s = 0; s < numSamples; s++)
	{
		float u = float(i+curand_uniform(&localRandState)) / float(maxX);
		float v = float(j+curand_uniform(&localRandState)) / float(maxY);
		Ray r = (*cam)->generateRay(u, v, &localRandState);
		outCol += colour(r, world,0, &localRandState);
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

__device__ inline void scene2(Hitable** dList, Hitable** dWorld, Camera** dCamera, int width, int height, curandState* randState,int nx,int ny,unsigned char* texData)
{
	curandState localRandState = *randState;
	Texture* checker = new CheckerTexture(new ConstantTexture(Vector3(.2f, .3f, .1f)),
		new ConstantTexture(Vector3(.9f, .9f, .9f)));
	Texture* earth = new ImageTexture(texData, nx, ny);
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
		30.0,
		float(width) / float(height),
		aperture,
		dist_to_focus,
		0.f,1.f);
}

//Select active scene here
__global__ void createWorld(Hitable** dList, Hitable** dWorld,Camera** dCamera,int width,int height,curandState* randState,int nx,int ny,unsigned char* texData)
{
	if (threadIdx.x == 0&&blockIdx.x == 0)
	{
		//scene1(dList, dWorld, dCamera, width, height, randState);
		scene2(dList, dWorld, dCamera, width, height, randState,nx,ny,texData);
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
	std::cerr<<"\nUsing "<<tx<<"x"<<ty<<" blocks";
	int res=width*height;
	size_t fbSize=res*sizeof(Vector3);

	//Allocate Textures
	int nx, ny, nn;
	unsigned char* texData = stbi_load("earthmap.jpg", &nx, &ny, &nn, 0);
	size_t texSize = nx * ny * nn * sizeof(unsigned char);
	checkCudaErrors(cudaMallocManaged((void**)&texData, texSize));

	//Allocate frame buffer
	Vector3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb,fbSize));

	//Allocate random state
	curandState* dRandState;
	checkCudaErrors(cudaMalloc((void**)&dRandState, res * sizeof(curandState)));
	curandState* dRandState2;
	checkCudaErrors(cudaMalloc((void**)&dRandState2, sizeof(curandState)));

	// we need a 2nd random state to be initialised for the world creation
	randInit<<<1, 1 >>>(dRandState2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create our world and Camera
	Hitable** dList;
	// 488 objects in scene1
	int numObjects = 3;
	checkCudaErrors(cudaMalloc((void**)&dList, numObjects * sizeof(Hitable*)));
	Hitable** dWorld;
	checkCudaErrors(cudaMalloc((void**)&dWorld, sizeof(Hitable*)));
	Camera** dCamera;
	checkCudaErrors(cudaMalloc((void**)&dCamera, sizeof(Camera*)));
	createWorld<<<1,1>>>(dList, dWorld, dCamera,width,height,dRandState2,nx,ny,texData);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start,stop;
	

	//Render the frame buffer
	dim3 blocks(width/tx+1,height/ty+1);
	dim3 threads(tx,ty);
	start = clock();
	renderInit<<<blocks,threads>>>(width,height,dRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	std::cerr << "\nRender init in " << ((double)(stop-start))/CLOCKS_PER_SEC << " seconds.\n";
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
	stbi_write_png("laptopcheck.png",width,height,3,pixels,width*3);
	
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
	checkCudaErrors(cudaFree(texData));
	return 0;
}