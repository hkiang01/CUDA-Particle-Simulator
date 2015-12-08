#ifndef particlesystem_h
#define particlesystem_h

#pragma once

#include <iostream>
#include <vector>
#include <stdlib.h>
#include "Constants.h"
#include "particle.h"

class particleSystem
{
public:
	particleSystem();
	particleSystem(unsigned int numParticles);
	~particleSystem();
	void printParticles();
	std::vector<particle> particleSystem::getParticlesVector();
	void gravitySerial(unsigned int simulationLength);
	double* particlesPosDoubleArray();
	double* particlesVelDoubleArray();
	void printPosDoubleArray(double* posFloatArray);
	void printVelDoubleArray(double* velFloatArray);

	std::vector<particle> particles;
	double* posDoubleArrayPtr;
	double* velDoubleArrayPtr;
	bool particlesPosDoubleArrayCalled;
	bool particlesVelDoubleArrayCalled;
};

#endif