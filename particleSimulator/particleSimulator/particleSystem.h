#ifndef particlesystem_h
#define particlesystem_h

#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>

#include "particle.h"
#include "Constants.h"


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
	
	//parallel consideration
	void gravityBoth(float3* positions, float3* velocities, float3* accelerations, unsigned int numRounds);
	bool isSame(float3* positions, float3* velocities, float3* accelerations);
	
	std::vector<particle> particles;
	float* posFloatArrayPtr;
	float* velFloatArrayPtr;
	float* accFloatArrayPtr;

	bool particlesPosFloatArrayCalled;
	bool particlesVelFloatArrayCalled;
	bool particlesAccFloatArrayCalled;
};

#endif