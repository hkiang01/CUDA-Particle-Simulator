#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

enum ParticleArray
{
	PARTICLESYS_POS,
	PARTICLESYS_VEL,
};

template <typename T>
class ParticleSystemCUDA
{
public:
	ParticleSystemCUDA(unsigned int numBodies,
		unsigned int blockSize);
	~ParticleSystemCUDA();

	void update(T deltaTime);

	T *getArray(ParticleArray array);
	void setArray(ParticleArray array, const T *data);
	unsigned int getNumBodies() const { return numbodies; }

protected:
	ParticleSystemCUDA() {}
	void _initialize(int numBodies);
	void _teardown();

	unsigned int numBodies;
	unsigned int blockSize;
	unsigned int currentRead;
	unsigned int currentWrite;

	T *hostPos[1];
	T *hostVel;

	T *devPos[2];
	T *devVel;
};

template <typename T>
ParticleSystemCUDA<T>::ParticleSystemCUDA(unsigned int nBodies,
	unsigned int blkSize)
	: numBodies(nBodies),
	blockSize(blkSize),
	currentRead(0),
	currentWrite(1)
{
	hostPos[0] = 0;
	hostVel = 0;

	_initialize(nBodies);
}

template <typename T>
ParticleSystemCUDA<T>::~ParticleSystemCUDA()
{
	_teardown();
	numBodies = 0;
}

template <typename T>
void ParticleSystemCUDA<T>::_initialize(int nBodies)
{
	numBodies = nBodies;

	unsigned int memSize = sizeof(T) * 4 * numBodies;

	hostPos[0] = new T[numBodies * 4];
	hostVel = new T[numBodies * 4];

	memset(hostPos[0], 0, memSize);
	memset(hostVel, 0, memSize);
}

template <typename T>
void ParticleSystemCUDA<T>::_teardown()
{
	delete[] hostPos[0];
	delete[] hostVel;
}

template <typename T>
void ParticleSystemCUDA<T>::update(T dt)
{
	//tempname<T>(); //caller to kernel in cuda file

	std::swap(currentRead, currentWrite);
}

// device particle array -> host
// returns new host particle array
template <typename T>
T *ParticleSystemCUDA<T>::getArray(ParticleArray array)
{
	T *hostdata = 0;
	T *devdata = 0;

	switch (array)
	{
	case PARTICLESYS_POS:
		hostdata = hostPos[0];
		devdata = devPos[currentRead];
		break;

	case PARTICLESYS_VEL:
		hostdata = hostVel;
		devdata = devVel;
		break;
	}

	cudaMemcpy(hostdata, devdata, numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost);

	return hostdata;
}

template <typename T>
void ParticleSystemCUDA<T>::setArray(ParticleArray array, const T *data)
{
	currentRead = 0;
	currentWrite = 1;

	switch (array)
	{
	case PARTICLESYS_POS:
		cudaMemcpy(devPos[currentRead], data, numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice);
		break;
	case PARTICLESYS_VEL:
		cudaMemcpy(devVel, data, numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice);
		break;
	}
}