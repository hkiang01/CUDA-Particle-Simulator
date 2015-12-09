
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

__constant__ float GRAVITY_CUDA = 0.066742f;

#define cudaCheck(stmt) do {													\
	cudaError_t err = stmt;														\
	if (err != cudaSuccess) {													\
		fprintf(stderr, "Failed to run stmt ", #stmt);							\
		fprintf(stderr, "Got CUDA error ... %s\n", cudaGetErrorString(err));	\
	}																			\
} while (0);

//calculate forces and resultant acceleration for a SINGLE particle due to physics interactions with ALL particles in system
//also updates positions and velocities
__global__
void gravityParallelKernel(float* positions, float* velocities, float* accelerations, unsigned int simulationLength) {

	//strategy: one thread (id) per particle

	__shared__ float3 particles_shared[BLOCK_SIZE];
	__shared__ float3 velocities_shared[BLOCK_SIZE];
	__shared__ float3 accelerations_shared[BLOCK_SIZE];

	float3 accel = { 0.0f, 0.0f, 0.0 };

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= NUM_PARTICLES) return;

	//load phase via float3 conversion
	float3 pos, vel, acc;
	pos.x = positions[3 * id];
	pos.y = positions[3 * id + 1];
	pos.z = positions[3 * id + 2];
	particles_shared[threadIdx.x] = pos;
	vel.x = positions[3 * id];
	vel.y = positions[3 * id + 1];
	vel.z = positions[3 * id + 2];
	velocities_shared[threadIdx.x] = vel;
	acc.x = positions[3 * id];
	acc.y = positions[3 * id + 1];
	acc.z = positions[3 * id + 2];
	accelerations_shared[threadIdx.x] = acc;
	__syncthreads();

	float3 curr = pos; //current position for given iteration
	for (unsigned int simCount = 0; simCount < simulationLength; simCount++) {
		//acc calculation phase
		for (unsigned i = 0; i < BLOCK_SIZE && i < NUM_PARTICLES; i++) { //all (other) particles
			float3 other = particles_shared[i];
			if (curr.x != other.x || curr.y != other.y || curr.z != other.z) { //don't affect own particle
				float3 ray = { curr.x - other.x, curr.y - other.y, curr.z - other.z };
				float dist = ray.x * ray.x + ray.y * ray.y + ray.z * ray.z;
				float xadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.x / (dist * dist * dist);
				float yadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.y / (dist * dist * dist);
				float zadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.z / (dist * dist * dist);
				atomicAdd(&(accel.x), xadd);
				atomicAdd(&(accel.y), yadd);
				atomicAdd(&(accel.z), zadd);
			}
		}
		__syncthreads();

		//update phase
		particles_shared[id].x += velocities_shared[id].x * EPOCH; //EPOCH is dt
		particles_shared[id].y += velocities_shared[id].y * EPOCH;
		particles_shared[id].z += velocities_shared[id].z * EPOCH;
		curr = particles_shared[id]; //for next iteration (update current position)

		velocities_shared[id].x += accelerations_shared[id].x * EPOCH; //EPOCH is dt
		velocities_shared[id].y += accelerations_shared[id].y * EPOCH;
		velocities_shared[id].z += accelerations_shared[id].z * EPOCH;

		accelerations_shared[id].x = accel.x; //EPOCH is dt
		accelerations_shared[id].y = accel.y;
		accelerations_shared[id].z = accel.z;
		__syncthreads();
	}

	//output phase via float conversion
	positions[3 * id] = particles_shared[id].x;
	positions[3 * id + 1] = particles_shared[id].y;
	positions[3 * id + 2] = particles_shared[id].z;
	velocities[3 * id] = velocities_shared[id].x;
	velocities[3 * id + 1] = velocities_shared[id].y;
	velocities[3 * id + 2] = velocities_shared[id].z;
	accelerations[3 * id] = accelerations_shared[id].x;
	accelerations[3 * id + 1] = accelerations_shared[id].y;
	accelerations[3 * id + 2] = accelerations_shared[id].z;
}

void gravityParallel(float* hostPositions, float* hostVelocities, float* hostAccelerations, unsigned int simulationLength) {
	//CUDA prep code
	float* devicePositions;
	float* deviceVelocities;
	float* deviceAccelerations;
	size_t size = NUM_PARTICLES * 3 * sizeof(float);

	cudaCheck(cudaSetDevice(0)); //choose GPU
	cudaCheck(cudaMalloc((void **)&devicePositions, size));
	cudaCheck(cudaMalloc((void **)&deviceVelocities, size));
	cudaCheck(cudaMalloc((void **)&deviceAccelerations, size));
	cudaCheck(cudaMemcpy(devicePositions, hostPositions, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceVelocities, hostVelocities, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceAccelerations, hostAccelerations, size, cudaMemcpyHostToDevice));
	dim3 dimGrid, dimBlock;
	dimGrid.x = (size - 1) / BLOCK_SIZE + 1;
	dimBlock.x = BLOCK_SIZE;
	gravityParallelKernel <<<dimGrid, dimBlock >>>(devicePositions, deviceVelocities, deviceAccelerations, simulationLength);
	cudaCheck(cudaDeviceSynchronize());
	cudaCheck(cudaMemcpy(hostPositions, devicePositions, size, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(hostVelocities, deviceVelocities, size, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(hostAccelerations, deviceAccelerations, size, cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(devicePositions));
	cudaCheck(cudaFree(deviceVelocities));
	cudaCheck(cudaFree(deviceAccelerations));
	
	return;
}

//print particles after a single round of serial and parallel to compare output and check correctness
void particleSystem::gravityBoth(float* positions, float* velocities, float* accelerations, unsigned int numRounds) {
	unsigned int round;
	for (round = 0; round < numRounds; round++) {
		
		//SERIAL PORTION
		this->gravitySerial(1); //execution phase
		this->printParticles(); //print phase

		//PARALLEL PORTION
		gravityParallel(positions, velocities, accelerations, 1); //execution phase
		printParticlcesArrays(positions, velocities, accelerations); //print phase
	}

	//CUDA cleanup code
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
