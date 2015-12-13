# ECE 408 Project
CUDA is awesome - Built using Visual Studio 2013


[![Particle Simulation - Gravity (CUDA)](http://img.youtube.com/vi/UakBtV319oY/0.jpg)](http://www.youtube.com/watch?v=UakBtV319oY)

Particle Simulation with Gravitational forces for 300 particles. 3D simulation visualized in 2D, view is into the z-axis. no depth of field or size perspective reflected with particles. Programmed in Visual Studio 2013 in CUDA-enabled C++, rendered in OpenGL. Final Project for ECE 408 - Applied Parallel Computing.

### Dependencies
CUDA (7.5 at time of writing) https://developer.nvidia.com/cuda-downloads

freeglut (imported via NuGet package manager)
NuGet Package Manager Command: <code>Install-Package nupengl.core</code>

## Configuration
See "Constants.h" file
- screen resolution
- serial vs parallel execution
- number of particles
- terminal mode or simulation mode
- simulation length (terminal mode)
- verbose or short output

### Contributors
Gabriella Quirini
Harrison Kiang
