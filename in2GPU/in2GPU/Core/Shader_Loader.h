#pragma once

#include "../Dependencies/glew/glew.h"
#include "../Dependencies/freeglut/freeglut.h"

namespace Core{
class Shader_Loader{
    private:
		GLuint vertex_shader, fragment_shader;
		unsigned int Shader_Loader::ReadShader(char *filename, GLenum shaderType);
		GLuint  CreateShader(GLenum shaderType, const char* source, int x, char* shaderName);

	public:
		
		Shader_Loader(void);
		~Shader_Loader(void);
		
		GLuint CreateProgram(char* VertexShaderFilename, char* FragmentShaderFilename);

};

}