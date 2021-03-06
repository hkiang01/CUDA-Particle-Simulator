#include "particleSystem.h"

particleSystem::particleSystem()
{
	srand((unsigned int)time(0));
	particlesPosFloatArrayCalled = false;
	particlesVelFloatArrayCalled = false;
	particlesAccFloatArrayCalled = false;
	systemIteration = 0;
}

particleSystem::particleSystem(unsigned int numParticles)
{
	v3 zero = { 0.0, 0.0, 0.0 };
	srand((unsigned int)time(0));
	unsigned int i;
	for (i = 0; i < numParticles; i++) {
		particle p = particle();
		p.setID(i);
		p.randomPosition(0.0, (float)(WORLD_DIM)/4);
		//p.randomVelocity(0.0, (float)MAX_VEL);
		p.setVelocity(zero);
		//p.randomAcceleration(0.0, (float)MAX_ACC);
		p.setAcceleration(zero);
		p.setMass((float)UNIVERSAL_MASS);
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

	std::vector<particle> temp; //store the old (the read source), particles - class member - is where the updates occur
	//reason: no particle should update unless all compuation for all particles are finished

	unsigned int counter = 0;
	while (counter < simulationLength) {
		temp = particles;
		for (unsigned int i = 0; i < temp.size(); i++) {
			v3 force = v3(0.0, 0.0, 0.0);
			for (unsigned int j = 0; j < temp.size(); j++) {
				if (i != j) { // force on i (it) by j (itt)
					v3 currRay = temp[i].getRay(temp[j]);
					if (SERIAL_DEBUG) {
						printf("ray (%u,%u); (%f,%f,%f)\n", i, j, currRay.x, currRay.y, currRay.z);
					}
					float dist = temp[i].getDistance(temp[j]);
					if (SERIAL_DEBUG) {
						printf("distance (%u,%u); %f\n", i, j, dist);
					}
					float mi = temp[i].getMass();
					float mj = temp[j].getMass();
					float xadd =(float)GRAVITY * (float)mj * (float)currRay.x / (float)pow(dist, 2.0);
					float yadd = (float)GRAVITY * (float)mj * (float)currRay.y / (float)pow(dist, 2.0);
					float zadd = (float)GRAVITY * (float)mj * (float)currRay.z / (float)pow(dist, 2.0);
					if (SERIAL_DEBUG) {
						printf("(xadd, yadd, zadd) (%u,%u); (%f,%f,%f)\n", i, j, xadd, yadd, zadd);
					}
					force.x -= xadd/float(mi); // F=ma --> a=F/m
					force.y -= yadd/float(mi);
					force.z -= zadd/float(mi);
				}
			}
			//it->updateParticle(EPOCH, force);
			particles[i].updateParticle(EPOCH, force); //KEY: update occurs at the class member particles vector
			if (SERIAL_UPDATE_OUTPUT && (i == 0 || i == 299)) {
				std::cout << "update (" << particles[i].id << "): ";
				particles[i].printProps();
			}
		}
		counter++;
	}
	systemIteration += simulationLength;
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

//comment out if CUDA doesn't work
bool particleSystem::isSame(float3* p, float3* v, float3* a) {
	bool retval = true;
	for (unsigned int i = 0; i < NUM_PARTICLES; i++) {
		v3 pos = particles[i].getPosition();
		v3 vel = particles[i].getVelocity();
		v3 acc = particles[i].getAcceleration();

		bool currSame = (pos.x == p[i].x && pos.y == p[i].y && pos.z == p[i].z
			&& vel.x == v[i].x && vel.y == v[i].y && v[i].z
			&& acc.x == a[i].x && acc.y == a[i].y && acc.z == a[i].z);

		if (retval && !currSame) { //first difference
			std::cout << "NOT the same, differences below..." << std::endl;
		}
		if (!currSame) {
			retval = false;
			std::cout << "(serial) ";
			particles[i].printProps();
			printf("(parallel) - id: %d\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", i / 3, p[i].x, p[i].y, p[i].z, v[i].x, v[i].y, v[i].z, a[i].x, a[i].y, a[i].z);
		}
	}
	if (retval) std::cout << "SAME\n" << std::endl;
	std::cout << std::endl;
	return retval;
}

particleSystem::~particleSystem()
{
	delete[] posFloatArrayPtr;
	delete[] velFloatArrayPtr;
	delete[] accFloatArrayPtr;
}
