#include "v3.h"

void v3::randomize()
{
	x = (double)rand() / (double)RAND_MAX;
	y = (double)rand() / (double)RAND_MAX;
	z = (double)rand() / (double)RAND_MAX;
}

v3::v3()
{
	randomize();
}

v3::v3(double xIn, double yIn, double zIn) {
	x = xIn;
	y = yIn;
	z = zIn;
}

v3::~v3()
{
}