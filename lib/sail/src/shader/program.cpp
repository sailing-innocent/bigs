/**
 * @file program.cpp
 * @brief The Implementation of OpenGL Shader program
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailCu/util/shader/program.h"
#include <iostream>

namespace sail {

ShaderProgram::ShaderProgram(std::string name) : m_name(name) {
	m_id = glCreateProgram();
	std::cout << "CREATING PROGRAM " << m_name << " " << m_id << std::endl;
	is_linked = false;
	is_compute = false;
}

ShaderProgram::~ShaderProgram() {
	std::cout << "DELETING PROGRAM " << m_name << std::endl;
	glDeleteProgram(m_id);
}

ShaderProgram* ShaderProgram::attach_shader(ShaderBase s) {
	std::cout << "ATTACHING SHADER " << s.get_name() << " TO PROGRAM " << m_name << std::endl;
	if (!is_compute) {
		glAttachShader(m_id, s.get_shad());
		if (s.get_name() == "compute") {
			is_compute = true;
		}
		m_shaders.push_back(s.get_shad());
	} else {
		std::cout << "ERROR: TRYING TO LINK A NON COMPUTE SHADER TO COMPUTE PROGRAM" << std::endl;
	}
	return this;
}

void ShaderProgram::link_program() {
	glLinkProgram(m_id);

	if (check_compile_errors(m_id, "PROGRAM", m_name.c_str())) {
		is_linked = true;
		std::cout << "PROGRAM " << m_name << " CORRECTLY LINKED" << std::endl;
		while (!m_shaders.empty()) {
			glDeleteShader(m_shaders.back());
			m_shaders.pop_back();
		}
	} else {
		std::cout << "PROGRAM " << m_name << " NOT LINKED" << std::endl;
	}
}

void ShaderProgram::use() const noexcept {
	if (is_linked) {
		// std::cout << "USING PROGRAM " << m_id << " " << m_name << std::endl;
		glUseProgram(m_id);
	} else {
		//std::cout << "ERROR: PROGRAM " << m_name << " NOT LINKED" << std::endl;
	}
}

// setters

void ShaderProgram::set_mat4(const string_view name, const glm::mat4& mat) const noexcept {
	glUniformMatrix4fv(glGetUniformLocation(m_id, name.data()), 1, GL_FALSE, glm::value_ptr(mat));
}

void ShaderProgram::set_bool(const string_view name, bool value) const noexcept {
	glUniform1i(glGetUniformLocation(m_id, name.data()), static_cast<int>(value));
}

void ShaderProgram::set_int(const string_view name, int value) const noexcept {
	glUniform1i(glGetUniformLocation(m_id, name.data()), value);
}

void ShaderProgram::set_float(const string_view name, float value) const noexcept {
	glUniform1f(glGetUniformLocation(m_id, name.data()), value);
}

void ShaderProgram::set_float4(const string_view name, float v0, float v1, float v2, float v3) const noexcept {
	glUniform4f(glGetUniformLocation(m_id, name.data()), v0, v1, v2, v3);
}

void ShaderProgram::set_vec3(const string_view name, const glm::vec3& vec) const noexcept {
	glUniform3fv(glGetUniformLocation(m_id, name.data()), 1, glm::value_ptr(vec));
}

void ShaderProgram::set_vec4(const string_view name, const glm::vec4& vec) const noexcept {
	glUniform4fv(glGetUniformLocation(m_id, name.data()), 1, glm::value_ptr(vec));
}

}// namespace sail