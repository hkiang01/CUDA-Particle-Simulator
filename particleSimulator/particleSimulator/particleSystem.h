#ifndef particlesystem_h
#define particlesystem_h

#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>

#include "Constants.h"
#include "particle.h"

class particleSystem
{
public:
	particleSystem();
	particleSystem(unsigned int numParticles);
	~particleSystem();
	void printParticles();
	void printParticlcesArrays(float* p, float* v, float* a);
	std::vector<particle> particleSystem::getParticlesVector();
	float* particlesPosfloatArray();
	float* particlesVelfloatArray();
	float* particlesAccfloatArray();
	void printPosFloatArray(float* posFloatArray);
	void printVelFloatArray(float* velFloatArray);
	void printAccFloatArray(float* accFloatArray);
	void gravitySerial(unsigned int simulationLength);
	void gravityParallel(float* positions, float* velocities, float* accelerations, unsigned int simulationLength);
	float3 gravityParallelKernel(float3 curPos, float* positions, unsigned int simulationLength);
	void gravityBoth(float* positions, float* velocities, float* accelerations, unsigned int numRounds);

	std::vector<particle> particles;
	float* posFloatArrayPtr;
	float* velFloatArrayPtr;
	float* accFloatArrayPtr;
	bool particlesPosFloatArrayCalled;
	bool particlesVelFloatArrayCalled;
	bool particlesAccFloatArrayCalled;
};

#endif