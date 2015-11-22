#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<vector>

#include "Core/Shader_Loader.h"
 
using namespace Core;

GLuint vertex_shader, fragment_shader, program;
GLuint vertex_array_object;

void renderScene(void) {
 
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glClearColor(1.0, 0.0, 0.0, 1.0);//clear red
 
   //use the created program
   glUseProgram(program);
   
   //draw 3 vertices as triangles
   glDrawArrays(GL_TRIANGLES, 0, 24);
   glutSwapBuffers();

   
}

void Init(){
	glEnable(GL_DEPTH_TEST);

	//load and compile shaders
	Core::Shader_Loader sh=Core::Shader_Loader::Shader_Loader();
	program=sh.CreateProgram("Shaders\\Vertex_Shader.glsl","Shaders\\Fragment_Shader.glsl");

	//generate the vertex array
	glGenVertexArrays(1, &vertex_array_object);
	glBindVertexArray(vertex_array_object);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
}

int main(int argc, char **argv) {
 
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(800, 600);
	glutCreateWindow("Drawing my first triangle");
 
	glewInit();
	
	Init();

	// register callbacks
	glutDisplayFunc(renderScene);
 
	glutMainLoop();
 
	return 0;
}