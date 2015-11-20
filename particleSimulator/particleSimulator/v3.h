#ifndef __v3_h__
#define __v3_h__

#include <math.h>
#include <stdlib.h>

#pragma once
class v3
{
public:

	double x;
	double y;
	double z;

	v3();
	v3(double xIn, double yIn, double zIn);
	void randomize();
	~v3();
};

#endif