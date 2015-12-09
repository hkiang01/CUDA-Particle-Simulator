#ifndef __v3_h__
#define __v3_h__

#include <math.h>
#include <stdlib.h>

#pragma once
class v3
{
public:

	float x;
	float y;
	float z;

	v3();
	v3(float xIn, float yIn, float zIn);
	void randomize();
	~v3();
};

#endif