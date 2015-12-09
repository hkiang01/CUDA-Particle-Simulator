
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

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

//calculate forces and resultant acceleration for a SINGLE particle
__global__
void gravityParallelKernel(float3 curr, float* positions, unsigned int simulationLength, float3 &accel) {

	//strategy: one thread per particle

	__shared__ float3 particles_shared[BLOCK_SIZE];
	accel = { 0.0f, 0.0f, 0.0 };

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= NUM_PARTICLES) return;

	//load phase
	float3 pos;
	pos.x = positions[3 * id];
	pos.y = positions[3 * id + 1];
	pos.z = positions[3 * id + 2];
	particles_shared[threadIdx.x] = pos;
	__syncthreads();

	unsigned int i;
	for (i = 0; i < BLOCK_SIZE && i < NUM_PARTICLES; i++) { //all particles
		float3 other = particles_shared[i];
		if (curr.x != other.x || curr.y != other.y || curr.z != other.z) { //don't affect own particle
			float3 ray = { curr.x - other.x, curr.y - other.y, curr.z - other.z };
			float dist = ray.x * ray.x + ray.y * ray.y + ray.z * ray.z;
			float xadd = GRAVITY * UNIVERSAL_MASS * (float)ray.x / (dist * dist * dist);
			float yadd = GRAVITY * UNIVERSAL_MASS * (float)ray.y / (dist * dist * dist);
			float zadd = GRAVITY * UNIVERSAL_MASS * (float)ray.z / (dist * dist * dist);
			atomicAdd(&(accel.x), xadd);
			atomicAdd(&(accel.y), yadd);
			atomicAdd(&(accel.z), zadd);
		}
	}
	__syncthreads();
}

__global__
void gravityParallel(float* position, float* velocity, float* acceleration, unsigned int simulationLength) {

	//strategy: one thread per particle

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= NUM_PARTICLES) return;

	//adjust positions and velocities
	int i = 3 * index;

	float3 curPos;
	curPos.x = position[3 * i];
	curPos.y = position[3 * i + 1];
	curPos.z = position[3 * i + 2];
	float3 accel;
	gravityParallelKernel(curPos, position, simulationLength, accel); //accel passed by reference

	__syncthreads();

	position[i] += velocity[i] * EPOCH; //EPOCH is dt
	position[i + 1] += velocity[i + 1] * EPOCH;
	position[i + 2] += velocity[i + 2] * EPOCH;

	velocity[i] += velocity[i] * EPOCH; //EPOCH is dt
	velocity[i + 1] += velocity[i + 1] * EPOCH;
	velocity[i + 2] += velocity[i + 2] * EPOCH;

	velocity[i] += accel.x;
	velocity[i + 1] += accel.y;
	velocity[i + 2] += accel.z;

	__syncthreads();
	return;
}
//print particles after a single round of serial and parallel to compare output and check correctness
void particleSystem::gravityBoth(float* positions, float* velocities, float* accelerations, unsigned int numRounds) {
	unsigned int round;
	for (round = 0; round < numRounds; round++) {
		//serial portion
		this->gravitySerial(1);
		this->printParticles();

		//parallel portion
		gravityParallel(positions, velocities, accelerations, 1);
		printParticlcesArrays(positions, velocities, accelerations);
	}
}

int main()
{

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaCheck(cudaDeviceReset());

	particleSystem parSys(NUM_PARTICLES);
	//parSys.printParticles();
	//parSys.gravitySerial(SIMULATION_LENGTH);
	float* pos = parSys.particlesPosfloatArray();
	float* vel = parSys.particlesVelfloatArray();
	float* acc = parSys.particlesAccfloatArray();
	std::cout << std::endl;
	//parSys.printPosfloatArray(pos);
	//parSys.printVelfloatArray(vel);
	parSys.gravityBoth(pos, vel, acc, SIMULATION_LENGTH);

	system("pause"); //see output of terminal
	return 0;
}
