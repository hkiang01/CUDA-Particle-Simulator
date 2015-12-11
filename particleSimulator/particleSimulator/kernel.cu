
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
		fprintf(stderr, "Failed to run stmt ", #stmt); 							\
		fprintf(stderr, "Got CUDA error ... %s\n", cudaGetErrorString(err)); 	\
	}																			\
} while (0);

__constant__ float GRAVITY_CUDA = 100.066742f; //KEEP THIS THE SAME AS ITS CONSTANTS_H COUNTERPART!!!

//calculate forces and resultant acceleration for a SINGLE particle due to physics interactions with ALL particles in system
//also updates positions and velocities
__global__
void gravityParallelKernel(float3* __restrict__ positions, float3* __restrict__ velocities, float3* __restrict__ accelerations, unsigned int simulationLength) {

	//strategy: one thread (id) per particle

	/*__shared__ float3 particles_shared[BLOCK_SIZE];
	__shared__ float3 velocities_shared[BLOCK_SIZE];
	__shared__ float3 accelerations_shared[BLOCK_SIZE];

	__shared__ float3 particles_temp[BLOCK_SIZE];
	__shared__ float3 velocities_temp[BLOCK_SIZE];
	__shared__ float3 accelerations_temp[BLOCK_SIZE];*/

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= NUM_PARTICLES) return;

	float3 temp_pos;
	float3 temp_vel;
	float3 temp_acc;
	float3 force;

	/*LOAD PHASE via float3 conversion
	float3 pos, vel, acc;
	pos.x = positions[3 * id];
	pos.y = positions[3 * id + 1];
	pos.z = positions[3 * id + 2];
	particles_shared[threadIdx.x] = pos;
	vel.x = velocities[3 * id];
	vel.y = velocities[3 * id + 1];
	vel.z = velocities[3 * id + 2];
	velocities_shared[threadIdx.x] = vel;
	acc.x = accelerations[3 * id];
	acc.y = accelerations[3 * id + 1];
	acc.z = accelerations[3 * id + 2];
	accelerations_shared[threadIdx.x] = acc;
	
	if (PARALLEL_DEBUG) {
		printf("import - id: %d\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", id, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z);
	}
	__syncthreads();*/

	//CALCULATION PHASE
	for (unsigned int simCount = 0; simCount < simulationLength; simCount++) 
	{
		temp_pos = positions[id];
		temp_vel = velocities[id];
		temp_acc = accelerations[id];

		force = { 0.0f, 0.0f, 0.0 };
		for (unsigned i = 0; i < NUM_PARTICLES; i++) //all (other) particles
		{
			if (id != i) //don't affect own particle
			{
				float3 other = positions[i];
				float3 ray = { temp_pos.x - other.x, temp_pos.y - other.y, temp_pos.z - other.z };
				/*if (PARALLEL_DEBUG) {
					printf("ray (%u,%u); (%f,%f,%f)\n", id, i, ray.x, ray.y, ray.z);
					}*/
				float dist = (temp_pos.x - other.x)*(temp_pos.x - other.x) + (temp_pos.y - other.y)*(temp_pos.y - other.y) + (temp_pos.z - other.z)*(temp_pos.z - other.z);
				dist = sqrt(dist);
				/*if (PARALLEL_DEBUG) {
					printf("distance (%u,%u); %f\n", id, i, dist);
					}*/
				float xadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.x / (dist * dist * dist);
				float yadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.y / (dist * dist * dist);
				float zadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.z / (dist * dist * dist);
				/*if (PARALLEL_DEBUG) {
					printf("(xadd, yadd, zadd) (%u,%u); (%f,%f,%f)\n", id, i, xadd, yadd, zadd);
					}*/

				force.x += xadd / UNIVERSAL_MASS;
				force.y += yadd / UNIVERSAL_MASS;
				force.z += zadd / UNIVERSAL_MASS;

			}
		}

		//update phase
		positions[id].x += temp_vel.x * EPOCH; //EPOCH is dt
		positions[id].y += temp_vel.y * EPOCH;
		positions[id].z += temp_vel.z * EPOCH;

		velocities[id].x += temp_acc.x * EPOCH; //EPOCH is dt
		velocities[id].y += temp_acc.y * EPOCH;
		velocities[id].z += temp_acc.z * EPOCH;

		//this is why that shit's important
		accelerations[id].x = force.x; //EPOCH is dt
		accelerations[id].y = force.y;
		accelerations[id].z = force.z;

			
			/*
		if (PARALLEL_UPDATE_OUTPUT) {
			printf("update (%d)\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", id, particles_shared[id].x, particles_shared[id].y, particles_shared[id].z,
				velocities_shared[id].x, velocities_shared[id].y, velocities_shared[id].z,
				accelerations_shared[id].x, accelerations_shared[id].y, accelerations_shared[id].z);
		}*/
		
		if (id == 0 || id == 299)
			printf("update (%d)\tpos: (%f, %f, %f)\n", id, positions[id].x, positions[id].y, positions[id].z);

		__syncthreads();
	}
}

void gravityParallel(float3* hostPositions, float3* hostVelocities, float3* hostAccelerations, unsigned int simulationLength) {
	//CUDA prep code
	float3* devicePositions;
	float3* deviceVelocities;
	float3* deviceAccelerations;
	size_t size = NUM_PARTICLES * sizeof(float3);

	cudaCheck(cudaSetDevice(0)); //choose GPU
	cudaCheck(cudaMalloc((void **)&devicePositions, size));
	cudaCheck(cudaMalloc((void **)&deviceVelocities, size));
	cudaCheck(cudaMalloc((void **)&deviceAccelerations, size));
	cudaCheck(cudaMemcpy(devicePositions, hostPositions, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceVelocities, hostVelocities, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceAccelerations, hostAccelerations, size, cudaMemcpyHostToDevice));
	dim3 dimGrid, dimBlock;
	dimGrid.x = (NUM_PARTICLES - 1) / BLOCK_SIZE + 1;
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

/*print particles after a single round of serial and parallel to compare output and check correctness
void particleSystem::gravityBoth(float* positions, float* velocities, float* accelerations, unsigned int numRounds) {
	unsigned int round;
	for (round = 0; round < numRounds; round++) {
		
		//SERIAL PORTION
		std::cout << "Serial round " << round << std::endl;
		this->gravitySerial(1); //execution phase
		//this->printParticles(); //print phase
		std::cout << std::endl;

		//PARALLEL PORTION
		std::cout << "Parallel round " << round << std::endl;
		gravityParallel(positions, velocities, accelerations, 1); //execution phase
		//printParticlcesArrays(positions, velocities, accelerations); //print phase
		std::cout << std::endl;

		if (SAME_CHECK) {
			this->isSame(positions, velocities, accelerations);
		}
	}

	//CUDA cleanup code
}*/

int main()
{

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaCheck(cudaDeviceReset());
	std::cout << "Initizing Particle System..." << std::endl;
	particleSystem parSys(NUM_PARTICLES);
	//parSys.printParticles();
	//parSys.gravitySerial(SIMULATION_LENGTH);
	float* p = parSys.particlesPosfloatArray();
	float* v = parSys.particlesVelfloatArray();
	float* a = parSys.particlesAccfloatArray();

	float3 positions[NUM_PARTICLES];
	float3 velocities[NUM_PARTICLES];
	float3 accelerations[NUM_PARTICLES];

	//convert to float3
	float3 pos, vel, acc;
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		pos.x = p[3 * i];
		pos.y = p[3 * i + 1];
		pos.z = p[3 * i + 2];
		positions[i] = pos;
		vel.x = v[3 * i];
		vel.y = v[3 * i + 1];
		vel.z = v[3 * i + 2];
		velocities[i] = vel;
		acc.x = a[3 * i];
		acc.y = a[3 * i + 1];
		acc.z = a[3 * i + 2];
		accelerations[i] = acc;
	}

	std::cout << std::endl;
	//parSys.printPosFloatArray(pos);
	//parSys.printVelFloatArray(vel);
	//parSys.printAccFloatArray(acc);
	//parSys.gravityBoth(pos, vel, acc, SIMULATION_LENGTH);

	parSys.gravitySerial(SIMULATION_LENGTH);
	printf("\n");
	gravityParallel(positions, velocities, accelerations, SIMULATION_LENGTH);

	system("pause"); //see output of terminal
	return 0;
}
