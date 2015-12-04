#include "vectors.h"
//#include "particlesystemcuda.h"


#define cudaCheck(stmt) do {													\
	cudaError_t err = stmt;														\
	if (err != cudaSuccess) {													\
		fprintf(stderr, "Failed to run stmt ", #stmt);							\
		fprintf(stderr, "Got CUDA error ... %s\n", cudaGetErrorString(err));	\
			}																			\
} while (0);

template <typename T>
struct DeviceData
{
	T *devPos[2];
	T *devVel;
};

template <typename T>
__device__ typename vec3<T>::Type
computeAccel(typename vec4<T>::Type curPos, typename vec4<T>::Type *positions)
{
	typename vec3<T>::Type accel = { 0.0f, 0.0f, 0.0f };

	// Harrison: figure out what goes here

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
	
	if (index >= nBodies)
		return;

	typename vec4<T>::Type position = oldPos[index];
	typename vec4<T>::Type velocity = vel[index];
	typename vec3<T>::Type accel = computeAccel<T>(position, oldPos);

	velocity.x += accel.x * dt;
	velocity.y += accel.y * dt;
	velocity.z += accel.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	newPos[index] = position;
	vel[index] = velocity;

}

template <typename T>
void systemStep(DeviceData<T> *devArrays, unsigned int curRead, float dt, unsigned int nBodies, int blockSize)
{
	int numBlocks = (nBodies - 1) / blockSize + 1;
	int sharedMemSize = blockSize * 4 * sizeof(T);

	interaction<T><<< numBlocks, blockSize, sharedMemSize >>>
		((typename vec4<T>::Type *)devArrays.devPos[1 - curRead],
		 (typename vec4<T>::Type *)devArrays.devPos[curRead],
		 (typename vec4<T>::Type *)devArrays.devVel,
		 nBodies, dt);

}

// Explicit specialization
template void systemStep<float>(DeviceData<float> *devArrays, unsigned int curRead, float dt, unsigned int nBodies, int blockSize);

