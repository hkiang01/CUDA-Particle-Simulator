
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include "particle.h"

#define cudaCheck(stmt) do {													\
	cudaError_t err = stmt;														\
	if (err != cudaSuccess) {													\
		fpritnf(stderr, "Failed to run stmt ", #stmt);							\
		fprintf(stderr, "Got CUDA error ... %s", cudaGetErrorString(err));		\
	}																			\
} while (0);

#define NUM_PARTICLES 2
#define WORLD_DIM 100
#define MAX_VEL 5
#define UNIVERSAL_MASS 55.00
#define EPOCH .001f
#define SIMULATION_LENGTH 2

const double GRAVITY = 0.066742;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	std::vector<particle> particles;
	int i;
	for (i = 0; i < NUM_PARTICLES; i++) {
		particle p = particle();
		p.setID(i);
		p.randomPosition(0.0, (float)WORLD_DIM);
		p.randomVelocity(0.0, (float)MAX_VEL);
		p.setMass((float)UNIVERSAL_MASS);
		particles.push_back(p);
	}

	std::vector<particle> particles_copy;

	printf("Initial configuration...\n");
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		it->printProps();
		particles_copy.push_back(*it);
	}
	printf("\n");

	printf("Initial configuration copy...\n");
	for (std::vector<particle>::iterator it = particles_copy.begin(); it != particles_copy.end(); ++it) {
		it->printProps();
	}
	printf("\n");

	int counter = 0;
	while (counter < SIMULATION_LENGTH) {
		for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
			v3 force = v3(0.0, 0.0, 0.0);
			for (std::vector<particle>::iterator itt = particles.begin(); itt != particles.end(); ++itt) {
				if (it != itt) {
					// force on i (it) by j (itt)
					v3 currRay = it->getRay(*itt);
					double dist = it->getDistance(*itt);
					double mi = it->getMass();
					double mj = itt->getMass();
					force.x += (double)GRAVITY * (double)mj * (double)currRay.x / (double)pow(dist, 3.0);
					force.y += (double)GRAVITY * (double)mj * (double)currRay.y / (double)pow(dist, 3.0);
					force.z += (double)GRAVITY * (double)mj * (double)currRay.z / (double)pow(dist, 3.0);
				}
			}
			it->updateParticle(EPOCH, force);
			it->printProps();
		}
		counter++;
	}

	system("pause"); //see output of terminal
	return 0;
}

cudaError_t gravityWithCuda(std::vector<particle> particles) {
	cudaError_t cudaStatus;
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}