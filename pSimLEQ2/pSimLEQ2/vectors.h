#include <algorithm>
#include "builtin_types.h"

#define RAND_MAX 0x7fff


template <typename T> struct vec3
{
	typedef float   Type;
}; // dummy
template <>           struct vec3<float>
{
	typedef float3  Type;
};
template <>           struct vec3<double>
{
	typedef double3 Type;
};

template <typename T> struct vec4
{
	typedef float   Type;
}; // dummy
template <>           struct vec4<float>
{
	typedef float4  Type;
};
template <>           struct vec4<double>
{
	typedef double4 Type;
};

inline float
dot(float3 v0, float3 v1)
{
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

template <typename T>
void randomize(T *pos, T *vel, float clusterScale, float velocityScale, int nBodies, bool vec4vel)
{
	float scale = clusterScale * std::max<float>(1.0f, nBodies / (1024.0f));
	float vscale = velocityScale * scale;

	int p = 0, v = 0, i = 0;

	while (i < nBodies)
	{
		float3 point;
		point.x = rand() / (float)RAND_MAX * 2 - 1;
		point.y = rand() / (float)RAND_MAX * 2 - 1;
		point.z = rand() / (float)RAND_MAX * 2 - 1;
		float lenSqr = dot(point, point);

		if (lenSqr > 1)
			continue;

		float3 velocity;
		velocity.x = rand() / (float)RAND_MAX * 2 - 1;
		velocity.y = rand() / (float)RAND_MAX * 2 - 1;
		velocity.z = rand() / (float)RAND_MAX * 2 - 1;
		lenSqr = dot(velocity, velocity);

		if (lenSqr > 1)
			continue;

		pos[p++] = point.x * scale; // pos.x
		pos[p++] = point.y * scale; // pos.y
		pos[p++] = point.z * scale; // pos.z
		pos[p++] = 1.0f; // mass

		vel[v++] = velocity.x * vscale; // pos.x
		vel[v++] = velocity.y * vscale; // pos.x
		vel[v++] = velocity.z * vscale; // pos.x

		if (vec4vel) vel[v++] = 1.0f; // inverse mass

		i++;
	}
}
