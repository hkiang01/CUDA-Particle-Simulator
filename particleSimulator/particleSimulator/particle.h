#ifndef _particle_h__
#define __particle_h__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "v3.h"

#pragma once
class particle
{
public:
	__host__ __device__ particle();
	__host__ __device__ particle(v3 posvec);
	__host__ __device__ particle(v3 posvec, double mass_in);
	__host__ __device__ particle(v3 posvec, v3 velvec, v3 accvec);
	__host__ __device__ int getID();
	__host__ __device__ void setID(int id_in);
	__host__ __device__ v3 getPosition();
	__host__ __device__ void setPosition(v3 posvec);
	__host__ __device__ void randomPosition(double minVal, double maxVal);
	__host__ __device__ v3 getVelocity();
	__host__ __device__ void setVelocity(v3 velvec);
	__host__ __device__ void randomVelocity(double minVal, double maxVal);
	__host__ __device__ v3 getAcceleration();
	__host__ __device__ void setAcceleration(v3 accvec);
	__host__ __device__ void randomAcceleration(double minVal, double maxVal);
	__host__ __device__ double getMass();
	__host__ __device__ void setMass(double mass_in);
	__host__ __device__ void randomMass(double minVal, double maxVal);
	__host__ __device__ void updateParticle(double dt);
	__host__ __device__ void updateParticle(double dt, v3 accvec);
	__host__ __device__ void applyForce(v3 forcevec);
	__host__ __device__ void printProps();
	__host__ __device__ ~particle();

	//between particles
	__host__ __device__ v3 getRay(particle const& p_other);
	__host__ __device__ double getDistance(particle const& other);

	int id;
	bool alive;
	double mass;
	v3 pos;
	v3 vel;
	v3 acc;
};

#endif