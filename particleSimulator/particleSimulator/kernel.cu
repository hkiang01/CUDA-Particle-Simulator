
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>

#include <iostream>
#include <stdio.h>
#include <vector>
#include "Constants.h"
#include "particle.h"
#include "particleSystem.h"

#define cudaCheck(stmt) do {													\
	cudaError_t err = stmt;														\
	if (err != cudaSuccess) {													\
		fprintf(stderr, "Failed to run stmt ", #stmt);							\
		fprintf(stderr, "Got CUDA error ... %s\n", cudaGetErrorString(err));	\
	}																			\
} while (0);

#define BLOCK_SIZE 256

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void gravitySerial(std::vector<particle> particles);
void gravityWithCuda(particle* particles, int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void gravKernel(particle* particles, int size, int simul_length) {
	
	//todo: kernel where each thread handles/updated a single particle
	//hard part: make this work across blocks like in mp 5.2 (not sure if this is doable)

	//__shared__ float particles_shared[BLOCK_SIZE];
	//above DOES NOT COMPILE (shared array of objects)
	//see Dynamic Shared Memory: http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
	//see response: http://stackoverflow.com/questions/27230621/cuda-shared-memory-inconsistent-results

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size) {
		//load phase
		//particles_shared[threadIdx.x] = particles[i];
		__syncthreads();

	}
}

int main()
{

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaCheck(cudaDeviceReset());

	particleSystem parSys(NUM_PARTICLES);
	parSys.printParticles();
	double* pos = parSys.particlesPosDoubleArray();
	double* vel = parSys.particlesVelDoubleArray();
	parSys.printPosDoubleArray(pos);
	parSys.printVelDoubleArray(vel);
	//parSys.gravitySerial(SIMULATION_LENGTH);


	system("pause"); //see output of terminal
	return 0;
}

void gravityWithCuda(particle *particles, int size) {
	particle *particles_device;

	cudaCheck(cudaSetDevice(0)); //choose which GPU to run on
	cudaCheck(cudaMalloc((void **)&particles_device, size * sizeof(particle)));
	cudaCheck(cudaMemcpy(particles_device, particles, size * sizeof(particle), cudaMemcpyHostToDevice));

	dim3 dimGrid;
	dim3 dimBlock;
	dimGrid.x = (size - 1) / BLOCK_SIZE + 1;
	dimBlock.x = BLOCK_SIZE;

	gravKernel<<<dimGrid,dimBlock,dimBlock.x*sizeof(particle)>>>(particles_device, size, SIMULATION_LENGTH);

	cudaCheck(cudaDeviceSynchronize());
	cudaCheck(cudaMemcpy(particles, particles_device, size * sizeof(particle), cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(particles_device));
	return;
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