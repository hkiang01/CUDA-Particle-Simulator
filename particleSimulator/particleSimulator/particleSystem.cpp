#include "particleSystem.h"

particleSystem::particleSystem()
{
	particlesPosDoubleArrayCalled = false;
	particlesVelDoubleArrayCalled = false;
}

particleSystem::particleSystem(unsigned int numParticles)
{
	unsigned int i;
	for (i = 0; i < numParticles; i++) {
		particle p = particle();
		p.randomPosition(0.0, (float)WORLD_DIM);
		p.randomVelocity(0.0, (float)MAX_VEL);
		p.setMass((float)UNIVERSAL_MASS);
		this->particles.push_back(p);
	}
}

void particleSystem::printParticles() {
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		it->printProps();
	}
}

std::vector<particle> particleSystem::getParticlesVector() {
	return particles;
}

void particleSystem::gravitySerial(unsigned int simulationLength) {
	unsigned int counter = 0;
	while (counter < simulationLength) {
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
			std::cout << "Distance from 0 to 1: " << it->getDistance(*(it++)) << std::endl;
		}
		counter++;
	}
}

double* particleSystem::particlesPosDoubleArray() {
	particlesPosDoubleArrayCalled = true;
	double* posArray = (double*)std::malloc(sizeof(double) * NUM_PARTICLES * 3); //x,y,z
	double* curParticle = posArray;
	unsigned int counter = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		*curParticle = it->getPosition().x;
		curParticle += sizeof(double);
		*curParticle = it->getPosition().y;
		curParticle += sizeof(double);
		*curParticle = it->getPosition().z;
		if (counter < NUM_PARTICLES - 1) curParticle += sizeof(double);
		counter++;
	}
	posDoubleArrayPtr = posArray;
	return posArray;
}

double* particleSystem::particlesVelDoubleArray() {
	particlesVelDoubleArrayCalled = true;
	double* velArray = (double*)std::malloc(sizeof(double) * NUM_PARTICLES * 3); //x,y,z
	double* curParticle = velArray;
	unsigned int counter = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		*curParticle = it->getVelocity().x;
		curParticle += sizeof(double);
		*curParticle = it->getVelocity().y;
		curParticle += sizeof(double);
		*curParticle = it->getVelocity().z;
		if (counter < NUM_PARTICLES - 1) curParticle += sizeof(double);
		counter++;
	}
	velDoubleArrayPtr = velArray;
	return velArray;
}

void particleSystem::printPosDoubleArray(double* posDoubleArray) {
	double x, y, z;
	unsigned int i;
	//shouldn't change the pointer position when control is handed back to caller
	double* origPosition = posDoubleArray;
	for (i = 0; i < NUM_PARTICLES; i++) {
		x = *posDoubleArray;
		posDoubleArray += sizeof(double);
		y = *posDoubleArray;
		posDoubleArray += sizeof(double);
		z = *posDoubleArray;
		posDoubleArray += sizeof(double);
		printf("id: %d\tpos: (%lf, %lf, %lf)\n", i, x, y, z);
	}
	posDoubleArray = origPosition;
}

void particleSystem::printVelDoubleArray(double* velDoubleArray) {
	double x, y, z;
	unsigned int i;
	//shouldn't change the pointer position when control is handed back to caller
	double* origPosition = velDoubleArray;
	for (i = 0; i < NUM_PARTICLES; i++) {
		x = *velDoubleArray;
		velDoubleArray += sizeof(double);
		y = *velDoubleArray;
		velDoubleArray += sizeof(double);
		z = *velDoubleArray;
		velDoubleArray += sizeof(double);
		printf("id: %d\tvel: (%lf, %lf, %lf)\n", i, x, y, z);
	}
	velDoubleArray = origPosition;
}

particleSystem::~particleSystem()
{
	//if (particlesPosDoubleArrayCalled) free(posDoubleArrayPtr);
	//if (particlesVelDoubleArrayCalled) free(velDoubleArrayPtr);
}
