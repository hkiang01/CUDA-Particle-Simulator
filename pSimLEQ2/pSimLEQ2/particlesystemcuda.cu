#include "vectors.h"
//#include "particlesystemcuda.h"

#define BLOCK_SIZE 256
#define GRAVITY 0.066742

#define cudaCheck(stmt) do {													\
	cudaError_t err = stmt;														\
	if (err != cudaSuccess) {													\
		fprintf(stderr, "Failed to run stmt ", #stmt);							\
		fprintf(stderr, "Got CUDA error ... %s\n", cudaGetErrorString(err));	\
			}																	\
} while (0);

template <typename T>
struct DeviceData
{
	T *devPos[2];
	T *devVel;
};

template <typename T>
__device__ typename vec4<T>::Type
computeAccel(typename vec4<T>::Type curPos, typename vec4<T>::Type *positions, unsigned int nBodies, double mass)
{
	// todo: multiple blocks case
	// assumption: all particles same mass (passed in)

	__shared__ float4 particles_shared[BLOCK_SIZE];
	typename vec4<T>::Type accel = { 0.0f, 0.0f, 0.0f, 0.0f };

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= nBodies) return accel;

	//load phase
	particles_shared[threadIdx.x] = positions[id];
	__syncthreads();

	unsigned int i;
	for (i = 0; i < BLOCK_SIZE && i < nBodies; i++) {
		typename vec4<T>::Type other = particles_shared[threadIdx.x];
		if (other.x != curPos.x || other.y != curPos.y || other.z != curPos.z) { //don't affect own particle
			typename vec3<T>::Type ray = { curPos.x - other.x, curPos.y - other.y, curPos.z - other.z };
			double dist = ray.x * ray.x + ray.y * ray.y + ray.z * ray.z;
			double xadd = GRAVITY * mass * (double)ray.x / (dist * dist * dist);
			double yadd = GRAVITY * mass * (double)ray.y / (dist * dist * dist);
			double zadd = GRAVITY * mass * (double)ray.z / (dist * dist * dist);
			atomicAdd(&(accel.x), xadd);
			atomicAdd(&(accel.y), yadd);
			atomicAdd(&(accel.z), zadd);
		}
	}
	__syncthreads();
	return accel;
}

// __restrict__ prevents pointer aliasing
template <typename T>
__global__ void interaction(typename vec4<T>::Type *__restrict__ newPos,
							typename vec4<T>::Type *__restrict__ oldPos,
							typename vec4<T>::Type *vel,
							unsigned int nBodies,
							float dt)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	printf("interaction index: %d\n", index); //this doesn't print???

	if (index >= nBodies)
		return;

	typename vec4<T>::Type position = oldPos[index];
	typename vec4<T>::Type velocity = vel[index];
	printf("interaction - before:\toldPos=(%f,%f,%f)\tvel=(%f,%f,%f)\n", position.x, position.y, position.z, velocity.x, velocity.y, velocity.z);
	typename vec4<T>::Type accel = computeAccel<T>(position, oldPos, nBodies, (double)5.00);

	velocity.x += accel.x * dt;
	velocity.y += accel.y * dt;
	velocity.z += accel.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	newPos[index] = position;
	vel[index] = velocity;

	printf("interaction - after:\tpos=(%f, %f, %f)\tvel=(%f, %f, %f)\tacc=(%f, %f, %f)\n", position.x, position.y, position.z, velocity.x, velocity.y, velocity.z, accel.x, accel.y, accel.z);

}

template <typename T>
void systemStep(DeviceData<T> *devArrays, unsigned int curRead, float dt, unsigned int nBodies, int blockSize)
{
	int numBlocks = (nBodies - 1) / blockSize + 1;
	int sharedMemSize = blockSize * 4 * sizeof(T);

	printf("systemStep\n");

	interaction<T><<< numBlocks, blockSize, sharedMemSize >>>
		((typename vec4<T>::Type *)devArrays->devPos[1 - curRead],
		 (typename vec4<T>::Type *)devArrays->devPos[curRead],
		 (typename vec4<T>::Type *)devArrays->devVel,
		 nBodies, dt);

}

// Explicit specialization
template void systemStep<float>(DeviceData<float> *devArrays, unsigned int curRead, float dt, unsigned int nBodies, int blockSize);

