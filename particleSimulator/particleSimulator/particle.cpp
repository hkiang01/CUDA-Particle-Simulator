#include "particle.h"


particle::particle()
{
	id = -1;
	alive = true;
	mass = 0.0;
	pos.x = 0.0;
	pos.y = 0.0;
	pos.z = 0.0;
	vel.x = 0.0;
	vel.y = 0.0;
	vel.z = 0.0;
}

particle::particle(v3 posvec) {
	this->setPosition(posvec);
}

particle::particle(v3 posvec, double mass_in) {
	this->setPosition(posvec);
	this->setMass(mass_in);
}

particle::particle(v3 posvec, v3 velvec, v3 accvec) {
	this->setPosition(posvec);
	this->setVelocity(velvec);
	this->setAcceleration(accvec);
}

__host__ __device__
void particle::setMass(double mass_in) {
	mass = mass_in;
}

__host__ __device__
void particle::randomMass(double minVal, double maxVal) {
	if (maxVal > minVal && minVal > 0.0) {
		double calculatedMass = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		this->setMass(calculatedMass);
	}
	else if (minVal < 0.0) {
		fprintf(stderr, "Error: Invalid params for randomMass - minVal must be greater than 0.0\n");
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomPosition - maxVal must be greater than minVal\n");
	}
}

__host__ __device__
double particle::getMass() {
	return mass;
}

__host__ __device__
void particle::setID(int id_in) {
	id = id_in;
}

__host__ __device__
int particle::getID() {
	return id;
}

__host__ __device__
v3 particle::getPosition() {
	return pos;
}

__host__ __device__
void particle::setPosition(v3 posvec) {
	pos.x = posvec.x;
	pos.y = posvec.y;
	pos.z = posvec.z;
}

__host__ __device__
void particle::randomPosition(double minVal, double maxVal) {
	if (maxVal > minVal) {
		double diff = maxVal - minVal;
		v3 posvec;
		posvec.x = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		posvec.y = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		posvec.z = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		this->setPosition(posvec);
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomPosition - maxVal must be greater than minVal\n");
	}
}

__host__ __device__
double particle::getDistance(particle const& other) {
	double xarg = this->pos.x - other.pos.x;
	double yarg = this->pos.y - other.pos.y;
	double zarg = this->pos.z - other.pos.z;
	double retval = pow(xarg, 2.0) + pow(yarg, 2.0) + pow(zarg, 2.0);
	return sqrt(retval);
}

__host__ __device__
v3 particle::getRay(particle const& p_other) {
	v3 retval;
	retval.x = this->pos.x - p_other.pos.x;
	retval.y = this->pos.y - p_other.pos.y;
	retval.z = this->pos.z - p_other.pos.z;
	return retval;
}

__host__ __device__
v3 particle::getVelocity() {
	return vel;
}

__host__ __device__
void particle::setVelocity(v3 velvec) {
	vel.x = velvec.x;
	vel.y = velvec.y;
	vel.z = velvec.z;
}

__host__ __device__
void particle::randomVelocity(double minVal, double maxVal) {
	if (maxVal > minVal) {
		double diff = maxVal - minVal;
		v3 velvec;
		velvec.x = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		velvec.y = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		velvec.z = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		this->setVelocity(velvec);
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomVelocity - maxVal must be greater than minVal\n");
	}
}

__host__ __device__
v3 particle::getAcceleration() {
	return acc;
}

__host__ __device__
void particle::setAcceleration(v3 accvec) {
	acc.x = accvec.x;
	acc.y = accvec.y;
	acc.z = accvec.z;
}

__host__ __device__
void particle::randomAcceleration(double minVal, double maxVal) {
	if (maxVal > minVal) {
		double diff = maxVal - minVal;
		v3 accvec;
		accvec.x = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		accvec.y = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		accvec.z = minVal + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (maxVal - minVal)));
		this->setAcceleration(accvec);
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomAcceleration - maxVal must be greater than minVal\n");
	}
}

__host__ __device__
void particle::updateParticle(double dt) {
	pos.x = pos.x + vel.x * dt;
	pos.y = pos.y + vel.y * dt;
	pos.z = pos.z + vel.z * dt;
	vel.x = vel.x + acc.x * dt;
	vel.y = vel.y + acc.y * dt;
	vel.z = vel.z + acc.z * dt;
}

__host__ __device__
void particle::updateParticle(double dt, v3 accvec) {
	pos.x = pos.x + vel.x * dt;
	pos.y = pos.y + vel.y * dt;
	pos.z = pos.z + vel.z * dt;
	vel.x = vel.x + acc.x * dt;
	vel.y = vel.y + acc.y * dt;
	vel.z = vel.z + acc.z * dt;
	this->setAcceleration(accvec);
}

__host__ __device__
void particle::applyForce(v3 forcevec) {
	v3 accvec = v3(forcevec.x / mass, forcevec.y / mass, forcevec.z / mass);
	this->setAcceleration(accvec);
}

__host__ __device__
void particle::printProps() {
	printf("id: %d\tpos: (%lf, %lf, %lf)\tvel: (%lf, %lf, %lf)\tacc:(%lf, %lf, %lf)\n", id, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z);
}

particle::~particle()
{
}