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
		p.setID(i);
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
	particlesPosDoubleArrayCalled = true; //for destructor
	double* posArray = new double[NUM_PARTICLES * 3];
	unsigned int i = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		posArray[i] = it->getPosition().x;
		posArray[i + 1] = it->getPosition().y;
		posArray[i + 2] = it->getPosition().z;
		i += 3;
	}
	posDoubleArrayPtr = posArray; //for destructor
	return posArray;
}

double* particleSystem::particlesVelDoubleArray() {
	particlesVelDoubleArrayCalled = true; //for destructor
	double* velArray = new double[NUM_PARTICLES * 3];
	unsigned int i = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		velArray[i] = it->getVelocity().x;
		velArray[i + 1] = it->getVelocity().y;
		velArray[i + 2] = it->getVelocity().z;
		i += 3;
	}
	velDoubleArrayPtr = velArray; //for destructor
	return velArray;
}

void particleSystem::printPosDoubleArray(double* posDoubleArray) {
	double x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i+=3) {
		x = posDoubleArray[i];
		y = posDoubleArray[i + 1];
		z = posDoubleArray[i + 2];
		printf("id: %d\tpos: (%lf, %lf, %lf)\n", i/3, x, y, z);
	}
}

void particleSystem::printVelDoubleArray(double* velDoubleArray) {
	double x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i += 3) {
		x = velDoubleArray[i];
		y = velDoubleArray[i + 1];
		z = velDoubleArray[i + 2];
		printf("id: %d\tvel: (%lf, %lf, %lf)\n", i / 3, x, y, z);
	}
}

particleSystem::~particleSystem()
{
	//if (particlesPosDoubleArrayCalled) free(posDoubleArrayPtr);
	//if (particlesVelDoubleArrayCalled) free(velDoubleArrayPtr);
}
