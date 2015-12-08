#include "particleSystem.h"

particleSystem::particleSystem()
{

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
	size_t numParticles = particles.size();
	double posArray[NUM_PARTICLES * 3]; //x,y,z
	double* curParticle = posArray;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		*curParticle = it->getPosition().x;
		curParticle += sizeof(double);
		*curParticle = it->getPosition().y;
		curParticle += sizeof(double);
		*curParticle = it->getPosition().z;
		curParticle += sizeof(double);
	}
	return posArray;
}

double* particleSystem::particlesVelDoubleArray() {
	size_t numParticles = particles.size();
	double velArray[NUM_PARTICLES * 3]; //x,y,z
	double* curParticle = velArray;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		*curParticle = it->getVelocity().x;
		curParticle += sizeof(double);
		*curParticle = it->getVelocity().y;
		curParticle += sizeof(double);
		*curParticle = it->getVelocity().z;
		curParticle += sizeof(double);
	}
	return velArray;
}

void particleSystem::printPosFloatArray(float* posFloatArray) {
	float x, y, z;
	unsigned int i;
	//shouldn't change the pointer position when control is handed back to caller
	float* origPosition = posFloatArray;
	for (i = 0; i < NUM_PARTICLES; i++) {
		x = *posFloatArray;
		posFloatArray += sizeof(float);
		y = *posFloatArray;
		posFloatArray += sizeof(float);
		z = *posFloatArray;
		posFloatArray += sizeof(float);
		printf("id: %d\tpos: (%lf, %lf, %lf)\n", i, x, y, z);
	}
	posFloatArray = origPosition;
}

void particleSystem::printVelFloatArray(float* velFloatArray) {
	float x, y, z;
	unsigned int i;
	//shouldn't change the pointer position when control is handed back to caller
	float* origPosition = velFloatArray;
	for (i = 0; i < NUM_PARTICLES; i++) {
		x = *velFloatArray;
		velFloatArray += sizeof(float);
		y = *velFloatArray;
		velFloatArray += sizeof(float);
		z = *velFloatArray;
		velFloatArray += sizeof(float);
		printf("id: %d\tvel: (%lf, %lf, %lf)\n", i, x, y, z);
	}
	velFloatArray = origPosition;
}

particleSystem::~particleSystem()
{
}
