#include "particleSystem.h"

particleSystem::particleSystem()
{
	particlesPosFloatArrayCalled = false;
	particlesVelFloatArrayCalled = false;
	particlesAccFloatArrayCalled = false;
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
		p.randomAcceleration(0.0, (float)MAX_ACC);
		this->particles.push_back(p);
	}
}

void particleSystem::printParticles() {
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		it->printProps();
	}
}

void particleSystem::printParticlcesArrays(float* p, float* v, float* a) {
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i += 3) {
		float3 pos, vel, acc;
		pos.x = p[i];
		pos.y = p[i + 1];
		pos.z = p[i + 2];
		vel.x = v[i];
		vel.y = v[i + 1];
		vel.z = v[i + 2];
		acc.x = a[i];
		acc.y = a[i + 1];
		acc.z = a[i + 2];
		printf("id: %d\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", i, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z);
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
					float dist = it->getDistance(*itt);
					float mi = it->getMass();
					float mj = itt->getMass();
					force.x += (float)GRAVITY * (float)mj * (float)currRay.x / (float)pow(dist, 3.0);
					force.y += (float)GRAVITY * (float)mj * (float)currRay.y / (float)pow(dist, 3.0);
					force.z += (float)GRAVITY * (float)mj * (float)currRay.z / (float)pow(dist, 3.0);
				}
			}
			it->updateParticle(EPOCH, force);
			//it->printProps();
			//std::cout << "Distance from 0 to 1: " << it->getDistance(*(it++)) << std::endl;
		}
		counter++;
	}
}


float* particleSystem::particlesPosfloatArray() {
	particlesPosFloatArrayCalled = true; //for destructor
	float* posArray = new float[NUM_PARTICLES * 3];
	unsigned int i = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		posArray[i] = it->getPosition().x;
		posArray[i + 1] = it->getPosition().y;
		posArray[i + 2] = it->getPosition().z;
		i += 3;
	}
	posFloatArrayPtr = posArray; //for destructor
	return posArray;
}

float* particleSystem::particlesVelfloatArray() {
	particlesVelFloatArrayCalled = true; //for destructor
	float* velArray = new float[NUM_PARTICLES * 3];
	unsigned int i = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		velArray[i] = it->getVelocity().x;
		velArray[i + 1] = it->getVelocity().y;
		velArray[i + 2] = it->getVelocity().z;
		i += 3;
	}
	velFloatArrayPtr = velArray; //for destructor
	return velArray;
}

float* particleSystem::particlesAccfloatArray() {
	particlesAccFloatArrayCalled = true; //for destructor
	float* accArray = new float[NUM_PARTICLES * 3];
	unsigned int i = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		accArray[i] = it->getVelocity().x;
		accArray[i + 1] = it->getVelocity().y;
		accArray[i + 2] = it->getVelocity().z;
		i += 3;
	}
	accFloatArrayPtr = accArray; //for destructor
	return accArray;
}

void particleSystem::printPosFloatArray(float* posfloatArray) {
	float x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i+=3) {
		x = posfloatArray[i];
		y = posfloatArray[i + 1];
		z = posfloatArray[i + 2];
		printf("id: %d\tpos: (%f, %f, %f)\n", i/3, x, y, z);
	}
}

void particleSystem::printVelFloatArray(float* velfloatArray) {
	float x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i += 3) {
		x = velfloatArray[i];
		y = velfloatArray[i + 1];
		z = velfloatArray[i + 2];
		printf("id: %d\tvel: (%f, %f, %f)\n", i / 3, x, y, z);
	}
}

particleSystem::~particleSystem()
{
	//if (particlesPosfloatArrayCalled) free(posfloatArrayPtr);
	//if (particlesVelfloatArrayCalled) free(velfloatArrayPtr);
	//if (particlesAccfloatArrayCalled) free(accfloatArrayPtr);
}
