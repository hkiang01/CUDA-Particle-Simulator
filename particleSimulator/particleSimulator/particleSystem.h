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
	void printParticlcesArrays(double* p, double* v, double* a);
	std::vector<particle> particleSystem::getParticlesVector();
	double* particlesPosDoubleArray();
	double* particlesVelDoubleArray();
	double* particlesAccDoubleArray();
	void printPosDoubleArray(double* posDoubleArray);
	void printVelDoubleArray(double* velDoubleArray);
	void printAccDoubleArray(double* accDoubleArray);
	void gravitySerial(unsigned int simulationLength);
	void gravityParallel(double* positions, double* velocities, double* accelerations, unsigned int simulationLength);
	double3 gravityParallelKernel(double3 curPos, double* positions, unsigned int simulationLength);
	void gravityBoth(double* positions, double* velocities, double* accelerations, unsigned int numRounds);

	std::vector<particle> particles;
	double* posDoubleArrayPtr;
	double* velDoubleArrayPtr;
	double* accDoubleArrayPtr;
	bool particlesPosDoubleArrayCalled;
	bool particlesVelDoubleArrayCalled;
	bool particlesAccDoubleArrayCalled;
};

#endif