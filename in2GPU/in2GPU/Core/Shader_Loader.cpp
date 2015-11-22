#include "Shader_Loader.h"
#include<iostream>
#include<fstream>
#include<vector>

namespace Core{
Shader_Loader::Shader_Loader(void){}

Shader_Loader::~Shader_Loader(void){}

unsigned int Shader_Loader::ReadShader(char *filename, GLenum shaderType){
	std::string shader_code;
	std::ifstream file(filename, std::ios::in);
	if(!file.good()){
		std::cout<<"Can't read"<<filename<<std::endl;
		std::terminate();
	}
	file.seekg(0, std::ios::end);
	shader_code.resize((unsigned int)file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(&shader_code[0], shader_code.size());
	file.close();

	int info_log_length = 0, compile_result = 0;
	unsigned int gl_shader_object;

	//construieste un obiect de tip shader din codul incarcat
	gl_shader_object = glCreateShader(shaderType);
	const char *shader_code_ptr = shader_code.c_str();
	const int shader_code_size = shader_code.size();
	glShaderSource(gl_shader_object, 1, &shader_code_ptr, &shader_code_size);
	glCompileShader(gl_shader_object);
	glGetShaderiv(gl_shader_object, GL_COMPILE_STATUS, &compile_result);

	//daca exista erori output la consola
	if (compile_result == GL_FALSE){
		std::string str_shader_type = "";
		if (shaderType == GL_VERTEX_SHADER) str_shader_type = "vertex shader";
		if (shaderType == GL_FRAGMENT_SHADER) str_shader_type = "fragment shader";

		glGetShaderiv(gl_shader_object, GL_INFO_LOG_LENGTH, &info_log_length);
		std::vector<char> shader_log(info_log_length);
		glGetShaderInfoLog(gl_shader_object, info_log_length, NULL, &shader_log[0]);
		std::cout << "Shader Loader: EROARE DE COMPILARE pentru " << std::endl << &shader_log[0] << std::endl;
		return 0;
	}

	return gl_shader_object;
}

GLuint Shader_Loader::CreateShader(GLenum shaderType, const char* source,int x, char* shaderName){

	GLenum shader = glCreateShader(shaderType);
	glShaderSource(shader, 1, &source, &x);
	glCompileShader(shader);

	GLint compile_result, info_log_length;

	glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_result);

	//check for errors in the shader
	if (compile_result == GL_FALSE){
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
		std::vector<char> shader_log(info_log_length);
		glGetShaderInfoLog(shader, info_log_length, NULL, &shader_log[0]);
		std::cout << "Shader Loader: Compilation error in "<< shaderName <<" shader" << std::endl << &shader_log[0] << std::endl;
	}
	return shader;
}

GLuint Shader_Loader::CreateProgram(char* vertexShaderFilename, char* fragmentShaderFilename){
	
      //read the shader files and save the content
	vertex_shader = ReadShader(vertexShaderFilename, GL_VERTEX_SHADER);
	fragment_shader = ReadShader(fragmentShaderFilename, GL_FRAGMENT_SHADER);

	

	int info_log_length = 0, link_result = 0;
	//create the program handle, attatch the shaders and link it
	GLuint program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);

	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_result);
	if (link_result == GL_FALSE){
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
		std::vector<char> program_log(info_log_length);
		glGetProgramInfoLog(program, info_log_length, NULL, &program_log[0]);
		std::cout << "Shader Loader : EROARE DE LINKARE" << std::endl << &program_log[0] << std::endl;
		return 0;
	}
	return program;
}
}