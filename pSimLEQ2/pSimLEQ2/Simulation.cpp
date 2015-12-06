#include <cstring>
#include <iostream>

using namespace std;

#include "particlesystemcuda.h"
#include "vectors.h"

#define BLOCKSIZE 256
#define NUM_ITERATIONS 2

int g_numBodies = 16384;

template <typename T>
class Simulation
{
public:
	static void Create()
	{
		sim = new Simulation;
	}
	static void Destroy()
	{
		delete sim;
	}
	static void init(int nBodies)
	{
		sim->_init(nBodies);
	}
	static void reset(int nBodies)
	{
		sim->_reset(nBodies);
	}
	static void run(int iterations)
	{
		sim->_run(iterations);
	}
	static void updateSim()
	{
		sim->particleSystem->update(timestep);
	}
	static void display(int numbodies)
	{
		for (int i = 0; i < numbodies; i++) {
			for (int j = i; j < i+4; j++) {
				switch (j-i) {
					case 0:
						cout << "pos[" << i << "](x,y,z,w): (" << sim->hostPos[j];
						break;
					case 1:
					case 2:
						cout << "," << sim->hostPos[j];
						break;
					case 3:
						cout << "," << sim << ")" << endl;
						break;
					default:
						break;
				}
			}
		}
		cout << endl;
		for (int i = 0; i < numbodies; i++) {
			for (int j = i; j < i + 4; j++) {
				switch (j-i) {
				case 0:
					cout << "vel[" << i << "](x,y,z,w): (" << sim->hostVel[j];
					break;
				case 1:
				case 2:
					cout << "," << sim->hostVel[j];
					break;
				case 3:
					cout << "," << sim << ")" << endl;
					break;
				default:
					break;
				}
			}
		}
		cout << endl;
	}
	static void getArrays()
	{
		T *_pos = sim->particleSystem->getArray(PARTICLESYS_POS);
		T *_vel = sim->particleSystem->getArray(PARTICLESYS_VEL);

		memcpy(sim->hostPos, _pos, sim->particleSystem->getNumBodies() * 4 * sizeof(T));
		memcpy(sim->hostVel, _vel, sim->particleSystem->getNumBodies() * 4 * sizeof(T));
	}
	static void setArrays(const T *pos, const T *vel)
	{
		if (pos != sim->hostPos)
			memcpy(sim->hostPos, pos, g_numBodies * 4 * sizeof(T));

		if (vel != sim->hostVel)
			memcpy(sim->hostVel, vel, g_numBodies * 4 * sizeof(T));

		sim->particleSystem->setArray(PARTICLESYS_POS, sim->hostPos);
		sim->particleSystem->setArray(PARTICLESYS_VEL, sim->hostVel);
	}

private:
	static Simulation *sim;
	ParticleSystemCUDA<T> *particleSystem;

	T *hostPos;
	T *hostVel;

	float timestep;
	float clusterscale;
	float velocityscale;

	Simulation() : particleSystem(0), hostPos(0), hostVel(0) {}

	~Simulation()
	{
		delete particleSystem;
		delete[] hostPos;
		delete[] hostVel;
	}

	void _init(int nBodies)
	{
		particleSystem = new ParticleSystemCUDA<T>(nBodies, BLOCKSIZE);

		hostPos = new T[nBodies * 4];
		hostVel = new T[nBodies * 4];

		timestep = 0.016f;		//between 0 and 1
		clusterscale = 1.52f;	//between 0 and 10
		velocityscale = 2.0f;	//between 0 and 1000

		// TODO init renderer
	}

	void _reset(int nBodies)
	{
		randomize(hostPos, hostVel, clusterscale, velocityscale, nBodies, true);
		setArrays(hostPos, hostVel);
	}

	void _run(int iterations)
	{
		particleSystem->update(timestep);

		for (int i = 0; i < iterations; ++i)
			particleSystem->update(timestep);
	}
};

template <> Simulation<float> *Simulation<float>::sim = 0;

void teardown()
{
	Simulation<float>::Destroy();
}

void display()
{
	//TODO
}


/********************
* MAIN
********************/
int main(int argc, char **argv)
{
	int numbodies = 2; //must be multiple of blocksize

	Simulation<float>::Create();
	Simulation<float>::init(numbodies);
	Simulation<float>::reset(numbodies);

	Simulation<float>::display(numbodies);

	Simulation<float>::run(NUM_ITERATIONS);

	Simulation<float>::getArrays();

	Simulation<float>::display(numbodies);

	system("pause");
	//teardown();

}