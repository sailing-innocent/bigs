/**
 * @file base.cpp
 * @brief The OpenGL Shader Base
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "SailCu/util/shader/base.h"
#include "SailCu/util/misc.h"
#include <iostream>

namespace sail {

ShaderBase ShaderBase::from_source(const char* shader_source, ShaderType type, std::string name) {
	ShaderBase shader;
	shader.m_shader_type = type;
	shader.m_shad = glCreateShader(type.m_type);
	glShaderSource(shader.m_shad, 1, &shader_source, NULL);
	glCompileShader(shader.m_shad);
	check_compile_errors(
		shader.m_shad, shader.m_shader_type.m_name, name);
	return shader;
}

ShaderBase ShaderBase::from_file(const char* shader_path) {
	auto path = std::string(shader_path);
	std::string shader_string = sail::read_file(shader_path);
	const char* shader_source = shader_string.c_str();
	return ShaderBase::from_source(shader_source, get_shader_type(shader_path), get_shader_name(shader_path));
}

ShaderBase::~ShaderBase() {}

bool check_compile_errors(unsigned int shader, std::string type_name, std::string shader_name) {
	int success;
	char infoLog[1024];
	if (type_name != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR: SHADER" << shader_name
					  << "COMPILATION ERROR of type: " << type_name << "\n"
					  << infoLog << "\n -- --------------------------------------------------- -- "
					  << std::endl;
		}
	} else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type_name << "\n"
					  << infoLog << "\n -- --------------------------------------------------- -- "
					  << std::endl;
		}
	}

	if (success) {
		std::cout << type_name + " SHADER SUCCESSFULLY COMPILED AND/OR LINKED!"
				  << std::endl;
	}
	return success;
}

std::string get_shader_name(const char* path) {
	std::string pathstr = std::string(path);
	const size_t last_slash_idx = pathstr.find_last_of("/");
	if (std::string::npos != last_slash_idx) {
		pathstr.erase(0, last_slash_idx + 1);
	}
	return pathstr;
}

ShaderType get_shader_type(const char* path) {
	std::string name = get_shader_name(path);
	const size_t last_slash_idx = name.find_last_of(".");
	if (std::string::npos != last_slash_idx) {
		name.erase(0, last_slash_idx + 1);
	}
	if (name == "vert") {
		return ShaderType(GL_VERTEX_SHADER, "VERTEX");
	}
	if (name == "frag") {
		return ShaderType(GL_FRAGMENT_SHADER, "FRAGMENT");
	}

	if (name == "tes") {
		return ShaderType(GL_TESS_EVALUATION_SHADER, "TESS_EVALUATION");
	}

	if (name == "tcs") {
		return ShaderType(GL_TESS_CONTROL_SHADER, "TESS_CONTROL");
	}

	if (name == "geom") {
		return ShaderType(GL_GEOMETRY_SHADER, "GEOMETRY");
	}

	if (name == "comp") {
		return ShaderType(GL_COMPUTE_SHADER, "COMPUTE");
	}
}

}// namespace sail
