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
		printf("id: %d\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", i/3, p[i], p[i + 1], p[i + 2], v[i], v[i + 1], v[i + 2], a[i], a[i + 1], a[i + 2]);
	}
}

std::vector<particle> particleSystem::getParticlesVector() {
	return particles;
}

void particleSystem::gravitySerial(unsigned int simulationLength) {

	if (SERIAL_DEBUG){
		for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
			std::cout << "import - ";
			it->printProps();
		}
	}
	unsigned int counter = 0;
	while (counter < simulationLength) {
		for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
			v3 force = v3(0.0, 0.0, 0.0);
			for (std::vector<particle>::iterator itt = particles.begin(); itt != particles.end(); ++itt) {
				if (it != itt) {
					// force on i (it) by j (itt)
					v3 currRay = it->getRay(*itt);
					if (SERIAL_DEBUG){ std::cout << "ray (" << it->id << "," << itt->id << "): (" << currRay.x << "," << currRay.y << "," << currRay.z << ")" << std::endl; }
					float dist = it->getDistance(*itt);
					if (SERIAL_DEBUG){ std::cout << "distance (" << it->id << "," << itt->id << "): " << dist << std::endl; }
					float mi = it->getMass();
					float mj = itt->getMass();
					float xadd =(float)GRAVITY * (float)mj * (float)currRay.x / (float)pow(dist, 3.0);
					float yadd = (float)GRAVITY * (float)mj * (float)currRay.y / (float)pow(dist, 3.0);
					float zadd = (float)GRAVITY * (float)mj * (float)currRay.z / (float)pow(dist, 3.0);
					if (SERIAL_DEBUG){ std::cout << "(xadd,yadd,zadd) (" << it->id << "," << itt->id << "): " << xadd << "," << yadd << "," << zadd << ")" << std::endl; }
					force.x += xadd/float(mi); // F=ma --> a=F/m
					force.y += yadd/float(mi);
					force.z += zadd/float(mi);
				}
			}
			it->updateParticle(EPOCH, force);
			if (SERIAL_UPDATE_OUTPUT) {
				std::cout << "update (" << it->id << "): ";
				it->printProps();
			}
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
		accArray[i] = it->getAcceleration().x;
		accArray[i + 1] = it->getAcceleration().y;
		accArray[i + 2] = it->getAcceleration().z;
		i += 3;
	}
	accFloatArrayPtr = accArray; //for destructor
	return accArray;
}

void particleSystem::printPosFloatArray(float* posFloatArray) {
	float x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i+=3) {
		x = posFloatArray[i];
		y = posFloatArray[i + 1];
		z = posFloatArray[i + 2];
		printf("id: %d\tpos: (%f, %f, %f)\n", i/3, x, y, z);
	}
}

void particleSystem::printVelFloatArray(float* velFloatArray) {
	float x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i += 3) {
		x = velFloatArray[i];
		y = velFloatArray[i + 1];
		z = velFloatArray[i + 2];
		printf("id: %d\tvel: (%f, %f, %f)\n", i / 3, x, y, z);
	}
}

void particleSystem::printAccFloatArray(float* accFloatArray) {
	float x, y, z;
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i += 3) {
		x = accFloatArray[i];
		y = accFloatArray[i + 1];
		z = accFloatArray[i + 2];
		printf("id: %d\tacc: (%f, %f, %f)\n", i / 3, x, y, z);
	}
}

particleSystem::~particleSystem()
{
	//if (particlesPosfloatArrayCalled) free(posfloatArrayPtr);
	//if (particlesVelfloatArrayCalled) free(velfloatArrayPtr);
	//if (particlesAccfloatArrayCalled) free(accfloatArrayPtr);
}
