#include "particleSystem.h"

particleSystem::particleSystem()
{
	particlesPosDoubleArrayCalled = false;
	particlesVelDoubleArrayCalled = false;
	particlesAccDoubleArrayCalled = false;
}

particleSystem::particleSystem(unsigned int numParticles)
{
	unsigned int i;
	for (i = 0; i < numParticles; i++) {
		particle p = particle();
		p.setID(i);
		p.randomPosition(0.0, (double)WORLD_DIM);
		p.randomVelocity(0.0, (double)MAX_VEL);
		p.setMass((double)UNIVERSAL_MASS);
		p.randomAcceleration(0.0, (double)MAX_ACC);
		this->particles.push_back(p);
	}
}

void particleSystem::printParticles() {
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		it->printProps();
	}
}

void particleSystem::printParticlcesArrays(double* p, double* v, double* a) {
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES * 3; i += 3) {
		double3 pos, vel, acc;
		pos.x = p[i];
		pos.y = p[i + 1];
		pos.z = p[i + 2];
		vel.x = v[i];
		vel.y = v[i + 1];
		vel.z = v[i + 2];
		acc.x = a[i];
		acc.y = a[i + 1];
		acc.z = a[i + 2];
		printf("id: %d\tpos: (%lf, %lf, %lf)\tvel: (%lf, %lf, %lf)\tacc:(%lf, %lf, %lf)\n", i, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z);
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
			//it->printProps();
			//std::cout << "Distance from 0 to 1: " << it->getDistance(*(it++)) << std::endl;
		}
		counter++;
	}
}

__global__
void particleSystem::gravityParallel(double* position, double* velocity, double* acceleration, unsigned int simulationLength) {
	
	//strategy: one thread per particle

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= NUM_PARTICLES) return;

	//adjust positions and velocities
	int i = 3 * index;
	
	double3 curPos;
	curPos.x = position[3 * i];
	curPos.y = position[3 * i + 1];
	curPos.z = position[3 * i + 2];
	double3 accel = this->gravityParallelKernel(curPos, position, simulationLength);

	position[i] += velocity[i] * EPOCH; //EPOCH is dt
	position[i + 1] += velocity[i + 1] * EPOCH;
	position[i + 2] += velocity[i + 2] * EPOCH;

	velocity[i] += velocity[i] * EPOCH; //EPOCH is dt
	velocity[i + 1] += velocity[i + 1] * EPOCH;
	velocity[i + 2] += velocity[i + 2] * EPOCH;

	velocity[i] += accel.x;
	velocity[i + 1] += accel.y;
	velocity[i + 2] += accel.z;
}

//calculate forces and resultant acceleration for a SINGLE particle
__global__
double3 particleSystem::gravityParallelKernel(double3 curr, double* positions, unsigned int simulationLength) {
	
	//strategy: one thread per particle
	
	__shared__ double3 particles_shared[BLOCK_SIZE];
	double3 accel = { 0.0, 0.0, 0.0 };

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= NUM_PARTICLES) return accel;

	//load phase
	double3 pos;
	pos.x = positions[3 * id];
	pos.y = positions[3 * id + 1];
	pos.z = positions[3 * id + 2];
	particles_shared[threadIdx.x] = pos;
	__syncthreads();

	unsigned int i;
	for (i = 0; i < BLOCK_SIZE && i < NUM_PARTICLES; i++) { //all particles
		double3 other = particles_shared[i];
		if (curr.x != other.x || curr.y != other.y || curr.z != other.z) { //don't affect own particle
			double3 ray = { curr.x - other.x, curr.y - other.y, curr.z - other.z };
			double dist = ray.x * ray.x + ray.y * ray.y + ray.z * ray.z;
			double xadd = GRAVITY * UNIVERSAL_MASS * (double)ray.x / (dist * dist * dist);
			double yadd = GRAVITY * UNIVERSAL_MASS * (double)ray.y / (dist * dist * dist);
			double zadd = GRAVITY * UNIVERSAL_MASS * (double)ray.z / (dist * dist * dist);
			atomicAdd(&(accel.x), xadd);
			atomicAdd(&(accel.y), yadd);
			atomicAdd(&(accel.z), zadd);
		}
	}
	return accel;
}

//print particles after a single round of serial and parallel to compare output and check correctness
void particleSystem::gravityBoth(double* positions, double* velocities, double* accelerations, unsigned int numRounds) {
	unsigned int round;
	for (round = 0; round < numRounds; round++) {
		//serial portion
		gravitySerial(1);
		printParticles();

		//parallel portion
		gravityParallel(positions, velocities, accelerations, 1);
		printParticlcesArrays(positions, velocities, accelerations);
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

double* particleSystem::particlesAccDoubleArray() {
	particlesAccDoubleArrayCalled = true; //for destructor
	double* accArray = new double[NUM_PARTICLES * 3];
	unsigned int i = 0;
	for (std::vector<particle>::iterator it = particles.begin(); it != particles.end(); ++it) {
		accArray[i] = it->getVelocity().x;
		accArray[i + 1] = it->getVelocity().y;
		accArray[i + 2] = it->getVelocity().z;
		i += 3;
	}
	accDoubleArrayPtr = accArray; //for destructor
	return accArray;
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
	//if (particlesAccDoubleArrayCalled) free(accDoubleArrayPtr);
}
