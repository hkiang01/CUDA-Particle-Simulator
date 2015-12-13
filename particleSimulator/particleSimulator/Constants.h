#ifndef constants_h
#define constants_h

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
//#include <cuda_gl_interop.h>
//#include <helper_cuda.h>


//Note: If GLUT not included in project, run "Install-Package nupengl.core" in NuGet PM Console

//adjust for your computer
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

//mode toggle
#define VISUAL_MODE false	//true for simulation, false for terminal output
#define VISUAL_PARALLEL true //true for GPU, false for CPU
#define TILE_MODE true

//particleSystem params
const float GRAVITY = (const float)6.67300E-9; //ENSURE CUDA COUNTERPART IS THE SAME (kernel.cu)
#define EPOCH 1000.0f //	this is dt	ENSURE CUDA COUNTERPART IS THE SAME (kernel.cu)

#define BLOCK_SIZE 128
#define NUM_PARTICLES 300 //reflected in terminal output mode
#define SIMULATION_LENGTH 5 //reflected in terminal output mode
#define WORLD_DIM 100
#define MAX_VEL 5
#define MAX_ACC 5
#define UNIVERSAL_MASS 55.00f

//debugging
#define SERIAL_DEBUG false
#define PARALLEL_DEBUG false
#define SERIAL_UPDATE_OUTPUT true //terminal output for each iteration for each particle
#define PARALLEL_UPDATE_OUTPUT true //terminal output for each iteration for each particle
#define SAME_CHECK false //pure debug

#endif