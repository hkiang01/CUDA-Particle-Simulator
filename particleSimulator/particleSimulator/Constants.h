#ifndef constants_h
#define constants_h

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

//particleSystem params
//const float GRAVITY = 100066.742f;
const float GRAVITY = (const float)6.67300E-3;
#define NUM_PARTICLES 300 //reflected in terminal output mode
#define WORLD_DIM 100
#define MAX_VEL 5
#define MAX_ACC 5
#define UNIVERSAL_MASS 55.00f
#define EPOCH 1000.0f
#define SIMULATION_LENGTH 10
#define BLOCK_SIZE 256

//debugging
#define SERIAL_DEBUG false
#define SERIAL_UPDATE_OUTPUT false
#define PARALLEL_DEBUG false
#define PARALLEL_UPDATE_OUTPUT false
#define SAME_CHECK false

//mode toggle
#define VISUAL_MODE false	//true for simulation, false for terminal output
#define VISUAL_PARALLEL false //true for GPU, false for CPU

//plotting


#endif