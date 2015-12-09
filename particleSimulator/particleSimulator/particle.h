#ifndef particle_h
#define particle_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "v3.h"
#include "Constants.h"

#pragma once
class particle
{
public:
	 particle();
	 particle(v3 posvec);
	 particle(v3 posvec, float mass_in);
	 particle(v3 posvec, v3 velvec, v3 accvec);
	 int getID();
	 void setID(int id_in);
	 v3 getPosition();
	 void setPosition(v3 posvec);
	 void randomPosition(float minVal, float maxVal);
	 v3 getVelocity();
	 void setVelocity(v3 velvec);
	 void randomVelocity(float minVal, float maxVal);
	 v3 getAcceleration();
	 void setAcceleration(v3 accvec);
	 void randomAcceleration(float minVal, float maxVal);
	 float getMass();
	 void setMass(float mass_in);
	 void randomMass(float minVal, float maxVal);
	 void updateParticle(float dt);
	 void updateParticle(float dt, v3 accvec);
	 void applyForce(v3 forcevec);
	 void printProps();
	 ~particle();

	//between particles
	 v3 getRay(particle const& p_other);
	 float getDistance(particle const& other);

	int id;
	bool alive;
	float mass;
	v3 pos;
	v3 vel;
	v3 acc;
};

#endif