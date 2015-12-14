#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <ctime>
#include "Constants.h"
#include "particle.h"
#include "particleSystem.h"
#include <GL/glut.h> // NuGet Package Manager Command: "Install-Package nupengl.core"

#define cudaCheck(stmt) do {													\
	cudaError_t err = stmt;														\
	if (err != cudaSuccess) {													\
		fprintf(stderr, "Failed to run stmt ", #stmt); 							\
		fprintf(stderr, "Got CUDA error ... %s\n", cudaGetErrorString(err)); 	\
	}																			\
} while (0);

__constant__ float GRAVITY_CUDA = 6.67300E-9; //KEEP THIS THE SAME AS ITS CONSTANTS_H COUNTERPART!!! (Constants.h)
__constant__ float EPOCH_CUDA = 1000.0f; //KEEP THIS THE SAME AS ITS CONSTANTS_H COUNTERPART!!! (Constants.h)

particleSystem* parSys;
std::clock_t start_time;
unsigned int parallel_iteration;

float3 positions[NUM_PARTICLES];
float3 velocities[NUM_PARTICLES];
float3 accelerations[NUM_PARTICLES];

float3 parallelPosBuffer[NUM_PARTICLES];
GLfloat xColor[NUM_PARTICLES];
GLfloat yColor[NUM_PARTICLES];
GLfloat zColor[NUM_PARTICLES];

__device__ float3
bodyBodyInteraction(float3 acc,
float3 pos, int id,
float3 other, int otherID)
{
	if (id == otherID) return acc;
	float3 r; //ray
	r.x = pos.x - other.x;
	r.y = pos.y - other.y;
	r.z = pos.z - other.z;
	if (PARALLEL_DEBUG) {
		printf("ray (%u,%u); (%f,%f,%f)\n", id, otherID, r.x, r.y, r.z);
	}
	float dist = r.x * r.x + r.y * r.y + r.z * r.z;
	dist = sqrt(dist);
	if (PARALLEL_DEBUG) {
		printf("distance (%u,%u); %f\n", id, otherID, dist);
	}
	float xadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)r.x / (dist * dist);
	float yadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)r.y / (dist * dist);
	float zadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)r.z / (dist * dist);
	if (PARALLEL_DEBUG) {
		printf("(xadd, yadd, zadd) (%u,%u); (%f,%f,%f)\n", id, otherID, xadd, yadd, zadd);
	}
	acc.x += xadd / UNIVERSAL_MASS;
	acc.y += yadd / UNIVERSAL_MASS;
	acc.z += zadd / UNIVERSAL_MASS;

	return acc;
}

//from reduction MP in ECE 408 / CS 483
__device__
void reductionFloat3(float3 * input, float3 output, int len) {
	//@@ Load a segment of the input vector into shared memory
	//@@ Traverse the reduction tree
	//@@ Write the computed sum of the block to the output vector at the 
	//@@ correct index

	__shared__ float partialSumX[2 * BLOCK_SIZE];
	__shared__ float partialSumY[2 * BLOCK_SIZE];
	__shared__ float partialSumZ[2 * BLOCK_SIZE];

	unsigned int tx = threadIdx.x;
	unsigned int start = 2 * blockIdx.x*blockDim.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	partialSumX[tx] = input[start + tx].x;
	partialSumX[blockDim.x + tx] = input[start + blockDim.x + tx].x;
	partialSumY[tx] = input[start + tx].x;
	partialSumY[blockDim.x + tx] = input[start + blockDim.x + tx].y;
	partialSumZ[tx] = input[start + tx].x;
	partialSumZ[blockDim.x + tx] = input[start + blockDim.x + tx].z;
	unsigned int stride;
	for (stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		if (tx % stride == 0 && (2 * i + stride) < (len)) {
			partialSumX[2 * tx] += partialSumX[2 * tx + stride];
			partialSumY[2 * tx] += partialSumY[2 * tx + stride];
			partialSumZ[2 * tx] += partialSumZ[2 * tx + stride];
		}
	}
	__syncthreads();
	if (tx == 0) {
		output.x += partialSumX[0];
		output.y += partialSumY[0];
		output.z += partialSumZ[0];
	}
}

//calculate forces and resultant acceleration for a SINGLE particle due to physics interactions with ALL particles in system
//also updates positions and velocities
__global__
void gravityParallelKernel(float3* __restrict__ positions, float3* __restrict__ velocities, float3* __restrict__ accelerations, unsigned int simulationLength, unsigned int numTiles) {

	//strategy: one thread (id) per particle

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= NUM_PARTICLES) return;

	float3 temp_pos;
	float3 temp_vel;
	float3 temp_acc;
	float3 force;
	__shared__ float3 positions_shared[BLOCK_SIZE]; //for TILE_MODE
	__shared__ float3 acc_update[BLOCK_SIZE]; //for TILE_REDUCTION_MODE
	__shared__ float3 totals[NUM_TILES];//for TILE_REDUCTION_MODE
	
	if (PARALLEL_DEBUG) {
		printf("import - id: %d\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", id, positions[id].x, positions[id].y, positions[id].z,
			velocities[id].x, velocities[id].y, velocities[id].z,
			accelerations[id].x, accelerations[id].y, accelerations[id].z);
	}

	//CALCULATION PHASE
	for (unsigned int simCount = 0; simCount < simulationLength; simCount++) 
	{
		
		temp_pos = positions[id];
		temp_vel = velocities[id];
		temp_acc = accelerations[id];
		force = { 0.0f, 0.0f, 0.0 };

		if (TILE_MODE) {
			for (int tile = 0; tile < numTiles; tile++)
			{
				positions_shared[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
				__syncthreads();
				if (TILE_REDUCTION_MODE) { //stores calculation of each pragma thread into shared memory that is summed by reduction kernel
					//THERE ARE BUGS HERE!!!!
					//THERE ARE BUGS HERE!!!!
					//THERE ARE BUGS HERE!!!!
					// This is the "tile_calculation"
					#pragma unroll 128
					for (unsigned int counter = 0; counter < blockDim.x; counter++)
					{
						if ((counter + tile*numTiles) >= NUM_PARTICLES) break;
						acc_update[counter] = bodyBodyInteraction(acc_update[counter], temp_pos, id, positions_shared[counter], counter + tile*blockDim.x);
					}
					__syncthreads();
					reductionFloat3(acc_update, totals[tile], fminf(TILE_SIZE, NUM_PARTICLES - (tile * NUM_TILES)));
					__syncthreads();
					//THERE ARE BUGS HERE!!!!
					//THERE ARE BUGS HERE!!!!
					//THERE ARE BUGS HERE!!!!
				}
				else {
					// This is the "tile_calculation"
					#pragma unroll 128
					for (unsigned int counter = 0; counter < blockDim.x; counter++)
					{
						if ((counter + tile*numTiles) >= NUM_PARTICLES) break;
						force = bodyBodyInteraction(force, temp_pos, id, positions_shared[counter], counter + tile*blockDim.x);
					}
				}
				__syncthreads();
			}
			if (TILE_REDUCTION_MODE) { //stores calculation of each pragma thread into shared memory that is summed by reduction kernel
				reductionFloat3(totals, force, NUM_BLOCKS);
				__syncthreads();
			}
		}
		else {
			//#pragma unroll 128
			for (unsigned i = 0; i < NUM_PARTICLES; i++) //all (other) particles
			{
				if (id != i) //don't affect own particle
				{
					float3 other = positions[i];
					float3 ray = { temp_pos.x - other.x, temp_pos.y - other.y, temp_pos.z - other.z };
					if (PARALLEL_DEBUG) {
						printf("ray (%u,%u); (%f,%f,%f)\n", id, i, ray.x, ray.y, ray.z);
					}
					float dist = (temp_pos.x - other.x)*(temp_pos.x - other.x) + (temp_pos.y - other.y)*(temp_pos.y - other.y) + (temp_pos.z - other.z)*(temp_pos.z - other.z);
					dist = sqrt(dist);
					if (PARALLEL_DEBUG) {
						printf("distance (%u,%u); %f\n", id, i, dist);
					}
					float xadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.x / (dist * dist);
					float yadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.y / (dist * dist);
					float zadd = GRAVITY_CUDA * UNIVERSAL_MASS * (float)ray.z / (dist * dist);
					if (PARALLEL_DEBUG) {
						printf("(xadd, yadd, zadd) (%u,%u); (%f,%f,%f)\n", id, i, xadd, yadd, zadd);
					}

					force.x += xadd / UNIVERSAL_MASS;
					force.y += yadd / UNIVERSAL_MASS;
					force.z += zadd / UNIVERSAL_MASS;
				}
			}
		}
		//update phase
		positions[id].x += temp_vel.x * EPOCH_CUDA; //EPOCH_CUDA is dt
		positions[id].y += temp_vel.y * EPOCH_CUDA;
		positions[id].z += temp_vel.z * EPOCH_CUDA;

		velocities[id].x += temp_acc.x * EPOCH_CUDA; //EPOCH_CUDA is dt
		velocities[id].y += temp_acc.y * EPOCH_CUDA;
		velocities[id].z += temp_acc.z * EPOCH_CUDA;

		//this is why that shit's important
		accelerations[id].x = -force.x; //EPOCH is dt
		accelerations[id].y = -force.y;
		accelerations[id].z = -force.z;
		/*
		if (PARALLEL_UPDATE_OUTPUT) {
			printf("update (%d)\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", id, positions[id].x, positions[id].y, positions[id].z,
				velocities[id].x, velocities[id].y, velocities[id].z,
				accelerations[id].x, accelerations[id].y, accelerations[id].z);
		}
		*/
		
		if (PARALLEL_UPDATE_OUTPUT && (id == 0 || id == 299))
			printf("update (%d)\tpos: (%f, %f, %f)\tvel: (%f, %f, %f)\tacc:(%f, %f, %f)\n", id, positions[id].x, positions[id].y, positions[id].z,
			velocities[id].x, velocities[id].y, velocities[id].z,
			accelerations[id].x, accelerations[id].y, accelerations[id].z);

		__syncthreads();
	}
}

void gravityParallel(float3* hostPositions, float3* hostVelocities, float3* hostAccelerations, unsigned int simulationLength) {
	//CUDA prep code
	float3* devicePositions;
	float3* deviceVelocities;
	float3* deviceAccelerations;
	size_t size = NUM_PARTICLES * sizeof(float3);

	cudaCheck(cudaSetDevice(0)); //choose GPU
	cudaCheck(cudaMalloc((void **)&devicePositions, size));
	cudaCheck(cudaMalloc((void **)&deviceVelocities, size));
	cudaCheck(cudaMalloc((void **)&deviceAccelerations, size));
	cudaCheck(cudaMemcpy(devicePositions, hostPositions, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceVelocities, hostVelocities, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceAccelerations, hostAccelerations, size, cudaMemcpyHostToDevice));
	dim3 dimGrid, dimBlock;
	dimGrid.x = (NUM_PARTICLES - 1) / BLOCK_SIZE + 1;
	dimBlock.x = BLOCK_SIZE;
	gravityParallelKernel <<<dimGrid, dimBlock >>>(devicePositions, deviceVelocities, deviceAccelerations, simulationLength, dimGrid.x);
	cudaCheck(cudaDeviceSynchronize());
	cudaCheck(cudaMemcpy(hostPositions, devicePositions, size, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(hostVelocities, deviceVelocities, size, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(hostAccelerations, deviceAccelerations, size, cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(devicePositions));
	cudaCheck(cudaFree(deviceVelocities));
	cudaCheck(cudaFree(deviceAccelerations));
	parallel_iteration+=simulationLength; //keep track of iterations
	return;
}

//print particles after a single round of serial and parallel to compare output and check correctness
void particleSystem::gravityBoth(float3* positions, float3* velocities, float3* accelerations, unsigned int numRounds) {
	unsigned int round;
	for (round = 0; round < numRounds; round++) {
		
		//SERIAL PORTION
		std::cout << "Serial round " << round << std::endl;
		this->gravitySerial(1); //execution phase
		systemIteration++;
		//this->printParticles(); //print phase
		std::cout << std::endl;

		//PARALLEL PORTION
		std::cout << "Parallel round " << round << std::endl;
		gravityParallel(positions, velocities, accelerations, 1); //execution phase
		memcpy(parallelPosBuffer, positions, NUM_PARTICLES*sizeof(float3));
		//printParticlcesArrays(positions, velocities, accelerations); //print phase
		std::cout << std::endl;

		if (SAME_CHECK) {
			this->isSame(positions, velocities, accelerations);
		}
	}

	//CUDA cleanup code
}
//Source (how to print text in OpenGL): http://www.codersource.net/2011/01/27/displaying-text-opengl-tutorial-5/
void drawBitmapText(char *string, size_t size, float x, float y, float z)
{
	char *c;
	glRasterPos3f(x, y, z);
	for (c = string; *c != '\0'; c++) glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c);
}

//Source (how to render basic OpenGL primitives): http://xoax.net/cpp/crs/opengl/lessons/
//lower left is (0,0) and upper right is (1,1) for XY 2D

void DrawSerial() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);

	//Frame and time
	char str[50] = "";
	double diff = (std::clock() - start_time) / (double)(CLOCKS_PER_SEC / 1000);
	//double fps = diff * 1000/ (double)parSys->systemIteration;
	//sprintf(str, "Frame: %u\tTime (ms): %u\tFPS: %lf", parSys->systemIteration, /*(unsigned int)*/diff, fps);
	sprintf(str, "Frame: %u\tTime (ms): %u", parSys->systemIteration, /*(unsigned int)*/diff);
	drawBitmapText(str, strlen(str), -WORLD_DIM, WORLD_DIM - 5.0, 0.0);

	//no need to loop as DrawSerial is called repeatedly, forever
	parSys->gravitySerial(1); //execution phase
	//what to draw
	unsigned int i;
	for (i = 0; i < NUM_PARTICLES; i++) {
		glPointSize((positions[i].z + WORLD_DIM)*GLfloat(3.0 / WORLD_DIM));
		glBegin(GL_POINTS);
		v3 pos = parSys->particles[i].getPosition();
		glColor3f(xColor[i], yColor[i], zColor[i]);
		glVertex3f(pos.x, pos.y, pos.z);
		glEnd();
	}
	glutSwapBuffers();
}

void DrawParallel() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);

	//Frame and time
	char str[50]= "";
	double diff = (std::clock() - start_time) / (double)(CLOCKS_PER_SEC / 1000);
	//double fps = diff * 1000 / (double)parallel_iteration;
	//sprintf(str, "Frame: %u\tTime (ms): %u\tFPS: %lf", parallel_iteration++, (unsigned int)diff, fps);
	sprintf(str, "Frame: %u\tTime (ms): %u", parallel_iteration++, (unsigned int)diff);
	drawBitmapText(str, strlen(str), -WORLD_DIM, WORLD_DIM - 5.0, 0.0);

	//no need to loop as DrawParallel is called repeatedly, forever
		gravityParallel(positions, velocities, accelerations, 1); //execution phase
		//what to draw
		unsigned int i;
		for (i = 0; i < NUM_PARTICLES; i++) {
			glPointSize((positions[i].z + WORLD_DIM)*GLfloat(3.0/WORLD_DIM));
			glBegin(GL_POINTS);
			glColor3f(xColor[i], yColor[i], zColor[i]);
			glVertex3f(positions[i].x, positions[i].y, positions[i].z);
			glEnd();
		}
	glutSwapBuffers();
}

//delayed animation
//https://youtu.be/Sl8FRfUy1ZA?t=218
void Timer(int iUnused) {
	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
}

void Initialize() {
	glClearColor(0.0, 0.0, 0.0, 0.0); //each range from 0 to 1 (0,0,0) is black
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-WORLD_DIM, WORLD_DIM, -WORLD_DIM, WORLD_DIM, -WORLD_DIM, WORLD_DIM); //x,y,z bounds
}

int main(int argc, char * argv[])
{

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaCheck(cudaDeviceReset());
	//std::cout << "Initizing Particle System..." << std::endl;
	parSys = new particleSystem(NUM_PARTICLES);
	//parSys.printParticles();
	//parSys.gravitySerial(SIMULATION_LENGTH);

	//get arrays from particleSystem object
	float* p = parSys->particlesPosfloatArray();
	float* v = parSys->particlesVelfloatArray();
	float* a = parSys->particlesAccfloatArray();

	//convert to float3
	float3 pos, vel, acc;
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		pos.x = p[3 * i];
		pos.y = p[3 * i + 1];
		pos.z = p[3 * i + 2];
		positions[i] = pos;
		vel.x = v[3 * i];
		vel.y = v[3 * i + 1];
		vel.z = v[3 * i + 2];
		velocities[i] = vel;
		acc.x = a[3 * i];
		acc.y = a[3 * i + 1];
		acc.z = a[3 * i + 2];
		accelerations[i] = acc;
		xColor[i] = (GLfloat)(rand() / ((float)RAND_MAX + 1));
		yColor[i] = (GLfloat)(rand() / ((float)RAND_MAX + 1));
		zColor[i] = (GLfloat)(rand() / ((float)RAND_MAX + 1));
	}
	/*
	std::cout << std::endl;
	parSys.printPosFloatArray(pos);
	parSys.printVelFloatArray(vel);
	parSys.printAccFloatArray(acc);
	*/

	//particleSystem instance already sets corresponding serial_iteration to 0
	parallel_iteration = 0;
	start_time = std::clock(); //reset start time

	//Visualization
	if (VISUAL_MODE) {
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
		glutInitWindowSize(1366, 768);
		glutInitWindowPosition(50, 40);
		glutCreateWindow("Particle Simulation Parallel");
		Initialize();
		if (VISUAL_PARALLEL){
			glutDisplayFunc(DrawParallel); //calls serial or parallel kernel
		}
		else {
			glutDisplayFunc(DrawSerial);
		}
		glutDisplayFunc(DrawParallel);
		Timer(0);
		glutMainLoop();
	}
	else {
		//parSys->gravityBoth(positions, velocities, accelerations, SIMULATION_LENGTH);
		
		//serial
		start_time = std::clock(); //reset start time
		parSys->gravitySerial(SIMULATION_LENGTH);
		double diff = (std::clock() - start_time) / (double)(CLOCKS_PER_SEC / 1000);
		printf("SERIAL: Time to run simulation of %u particles for length %u:\t\t%u ms\n", NUM_PARTICLES, SIMULATION_LENGTH, (unsigned int)diff);
		
		//parallel
		start_time = std::clock(); //reset start time
		gravityParallel(positions, velocities, accelerations, SIMULATION_LENGTH);
		diff = (std::clock() - start_time) / (double)(CLOCKS_PER_SEC / 1000);
		printf("PARALLEL: Time to run simulation of %u particles for length %u:\t\t%u ms\n", NUM_PARTICLES, SIMULATION_LENGTH, (unsigned int)diff);
	}

	delete[] p;
	delete[] v;
	delete[] a;
	system("pause"); //see output of terminal

	return 0;
}
