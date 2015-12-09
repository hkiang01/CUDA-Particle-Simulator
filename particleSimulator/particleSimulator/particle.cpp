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

 particle::particle(v3 posvec, float mass_in) {
	this->setPosition(posvec);
	this->setMass(mass_in);
}

 particle::particle(v3 posvec, v3 velvec, v3 accvec) {
	this->setPosition(posvec);
	this->setVelocity(velvec);
	this->setAcceleration(accvec);
}


void particle::setMass(float mass_in) {
	mass = mass_in;
}


void particle::randomMass(float minVal, float maxVal) {
	if (maxVal > minVal && minVal > 0.0) {
		float calculatedMass = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		this->setMass(calculatedMass);
	}
	else if (minVal < 0.0) {
		fprintf(stderr, "Error: Invalid params for randomMass - minVal must be greater than 0.0\n");
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomPosition - maxVal must be greater than minVal\n");
	}
}


float particle::getMass() {
	return mass;
}


void particle::setID(int id_in) {
	id = id_in;
}


int particle::getID() {
	return id;
}


v3 particle::getPosition() {
	return pos;
}


void particle::setPosition(v3 posvec) {
	pos.x = posvec.x;
	pos.y = posvec.y;
	pos.z = posvec.z;
}


void particle::randomPosition(float minVal, float maxVal) {
	if (maxVal > minVal) {
		float diff = maxVal - minVal;
		v3 posvec;
		posvec.x = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		posvec.y = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		posvec.z = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		this->setPosition(posvec);
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomPosition - maxVal must be greater than minVal\n");
	}
}


float particle::getDistance(particle const& other) {
	float xarg = this->pos.x - other.pos.x;
	float yarg = this->pos.y - other.pos.y;
	float zarg = this->pos.z - other.pos.z;
	float retval = pow(xarg, 2.0f) + pow(yarg, 2.0f) + pow(zarg, 2.0f);
	return sqrt(retval);
}


v3 particle::getRay(particle const& p_other) {
	v3 retval;
	retval.x = this->pos.x - p_other.pos.x;
	retval.y = this->pos.y - p_other.pos.y;
	retval.z = this->pos.z - p_other.pos.z;
	return retval;
}


v3 particle::getVelocity() {
	return vel;
}


void particle::setVelocity(v3 velvec) {
	vel.x = velvec.x;
	vel.y = velvec.y;
	vel.z = velvec.z;
}


void particle::randomVelocity(float minVal, float maxVal) {
	if (maxVal > minVal) {
		float diff = maxVal - minVal;
		v3 velvec;
		velvec.x = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		velvec.y = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		velvec.z = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		this->setVelocity(velvec);
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomVelocity - maxVal must be greater than minVal\n");
	}
}


v3 particle::getAcceleration() {
	return acc;
}


void particle::setAcceleration(v3 accvec) {
	acc.x = accvec.x;
	acc.y = accvec.y;
	acc.z = accvec.z;
}


void particle::randomAcceleration(float minVal, float maxVal) {
	if (maxVal > minVal) {
		float diff = maxVal - minVal;
		v3 accvec;
		accvec.x = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		accvec.y = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		accvec.z = minVal + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (maxVal - minVal)));
		this->setAcceleration(accvec);
	}
	else {
		fprintf(stderr, "Error: Invalid params for randomAcceleration - maxVal must be greater than minVal\n");
	}
}


void particle::updateParticle(float dt) {
	pos.x = pos.x + vel.x * dt;
	pos.y = pos.y + vel.y * dt;
	pos.z = pos.z + vel.z * dt;
	vel.x = vel.x + acc.x * dt;
	vel.y = vel.y + acc.y * dt;
	vel.z = vel.z + acc.z * dt;
}


void particle::updateParticle(float dt, v3 accvec) {
	pos.x = pos.x + vel.x * dt;
	pos.y = pos.y + vel.y * dt;
	pos.z = pos.z + vel.z * dt;
	vel.x = vel.x + acc.x * dt;
	vel.y = vel.y + acc.y * dt;
	vel.z = vel.z + acc.z * dt;
	this->setAcceleration(accvec);
}


void particle::applyForce(v3 forcevec) {
	v3 accvec = v3(forcevec.x / mass, forcevec.y / mass, forcevec.z / mass);
	this->setAcceleration(accvec);
}


void particle::printProps() {
	printf("id: %d\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", id, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, acc.x, acc.y, acc.z);
}


particle::~particle()
{
}