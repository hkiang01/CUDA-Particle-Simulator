#pragma once
// Tutorial: http://in2gpu.com/2014/10/29/shaders-basics/
// NuGet Package Manager Command: "Install-Package nupengl.core"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
namespace Core
{

	class Shader_Loader
	{
	private:

		std::string ReadShader(char *filename);
		GLuint CreateShader(GLenum shaderType,
			std::string source,
			char* shaderName);

	public:

		Shader_Loader(void);
		~Shader_Loader(void);
		GLuint CreateProgram(char* VertexShaderFilename,
			char* FragmentShaderFilename);

	};
}
